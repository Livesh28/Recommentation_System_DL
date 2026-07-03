import os
import numpy as np
import pandas as pd
import faiss
import joblib
from sentence_transformers import SentenceTransformer

class RecommenderEngine:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        
        # 1. Load products database
        products_path = os.path.join(self.data_dir, 'processed_products.csv')
        if not os.path.exists(products_path):
            raise FileNotFoundError(f"Missing '{products_path}'. Please run bootstrap.py first.")
        self.products_df = pd.read_csv(products_path)
        # Convert IDs to string for consistency
        self.products_df['product_id'] = self.products_df['product_id'].astype(str)
        # Fill NaN values to prevent JSON serialization and sorting errors
        for col in self.products_df.columns:
            if pd.api.types.is_numeric_dtype(self.products_df[col]):
                self.products_df[col] = self.products_df[col].fillna(0.0)
            else:
                self.products_df[col] = self.products_df[col].fillna('None')
        
        # 2. Load product embeddings
        embeddings_path = os.path.join(self.data_dir, 'product_embeddings.npy')
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Missing '{embeddings_path}'. Please run bootstrap.py first.")
        self.embeddings = np.load(embeddings_path)
        
        # 3. Load FAISS index
        faiss_path = os.path.join(self.data_dir, 'faiss_product_index.bin')
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"Missing '{faiss_path}'. Please run bootstrap.py first.")
        self.faiss_index = faiss.read_index(faiss_path)
        
        # 4. Load trained ML models
        preprocessor_path = os.path.join(self.data_dir, 'preprocessor.joblib')
        features_path = os.path.join(self.data_dir, 'feature_columns.joblib')
        eco_model_path = os.path.join(self.data_dir, 'eco_model.joblib')
        carbon_model_path = os.path.join(self.data_dir, 'carbon_credit_model.joblib')
        
        if not all(os.path.exists(p) for p in [preprocessor_path, features_path, eco_model_path, carbon_model_path]):
            raise FileNotFoundError("Missing preprocessor or predictor models. Please run bootstrap.py first.")
            
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_columns = joblib.load(features_path)
        self.eco_model = joblib.load(eco_model_path)
        self.carbon_model = joblib.load(carbon_model_path)
        
        # 5. Load interactions database
        self.interactions_path = os.path.join(self.data_dir, 'interactions.csv')
        self.load_interactions()
        
        # 6. Initialize embedding model (cache loaded locally)
        print("Initializing SentenceTransformer...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 7. Default hybrid scoring weights
        self.default_weights = {
            "semantic_similarity": 0.40,
            "personalized_ranking": 0.30,
            "sustainability_score": 0.30
        }

    def load_interactions(self):
        if os.path.exists(self.interactions_path):
            self.interactions_df = pd.read_csv(self.interactions_path)
            self.interactions_df['user_id'] = self.interactions_df['user_id'].astype(str)
            self.interactions_df['product_id'] = self.interactions_df['product_id'].astype(str)
        else:
            self.interactions_df = pd.DataFrame(columns=['user_id', 'product_id', 'interaction_score'])

    def add_interaction(self, user_id, product_id, score):
        user_id = str(user_id)
        product_id = str(product_id)
        score = int(score)
        
        # Check if interaction exists, if so update it, else append
        mask = (self.interactions_df['user_id'] == user_id) & (self.interactions_df['product_id'] == product_id)
        if mask.any():
            self.interactions_df.loc[mask, 'interaction_score'] = score
        else:
            new_row = pd.DataFrame([{'user_id': user_id, 'product_id': product_id, 'interaction_score': score}])
            self.interactions_df = pd.concat([self.interactions_df, new_row], ignore_index=True)
            
        # Write back to CSV
        self.interactions_df.to_csv(self.interactions_path, index=False)
        print(f"Added interaction: User={user_id}, Product={product_id}, Score={score}")

    def get_user_profile_embedding(self, user_id):
        user_id = str(user_id)
        user_int = self.interactions_df[(self.interactions_df['user_id'] == user_id) & (self.interactions_df['interaction_score'] >= 3)]
        if user_int.empty:
            return None
            
        profile_embeds = []
        weights = []
        for _, row in user_int.iterrows():
            pid = str(row['product_id'])
            # Find the index of this product in products_df
            p_idx = self.products_df[self.products_df['product_id'] == pid].index
            if len(p_idx) > 0:
                idx = p_idx[0]
                profile_embeds.append(self.embeddings[idx])
                # Weight by interaction score (1-5 scaled to 0.2-1.0)
                weights.append(row['interaction_score'] / 5.0)
                
        if not profile_embeds:
            return None
            
        # Weighted average of product embeddings
        profile_embeds = np.array(profile_embeds)
        weights = np.array(weights).reshape(-1, 1)
        weighted_embed = np.sum(profile_embeds * weights, axis=0) / np.sum(weights)
        return weighted_embed.astype('float32')

    def search_products(self, query, k=5):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        query_embedding = query_embedding.astype('float32')
        
        # FAISS search
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Compile results
        results = self.products_df.iloc[indices[0]].copy()
        
        # Calculate similarity score from distances (L2 distance -> similarity)
        # Cosine similarity on normalized embeddings is standard, but since FAISS uses FlatL2:
        # Distance = ||u - v||^2. We map this to [0, 1] range.
        max_dist = max(distances[0]) if len(distances[0]) > 0 else 1.0
        if max_dist > 0:
            results['semantic_similarity_score'] = 1.0 - (distances[0] / (max_dist * 1.5))
        else:
            results['semantic_similarity_score'] = 1.0
            
        results['semantic_similarity_score'] = results['semantic_similarity_score'].clip(0.0, 1.0)
        return results

    def recommend_similar(self, product_id, k=5):
        product_id = str(product_id)
        p_idx = self.products_df[self.products_df['product_id'] == product_id].index
        if p_idx.empty:
            return pd.DataFrame()
            
        idx = p_idx[0]
        prod_embedding = self.embeddings[idx].reshape(1, -1).astype('float32')
        
        # Query k + 1 because the product itself will be returned
        distances, indices = self.faiss_index.search(prod_embedding, k + 1)
        
        similar_indices = [i for i in indices[0] if i != idx][:k]
        similar_dists = [distances[0][j] for j, i in enumerate(indices[0]) if i != idx][:k]
        
        results = self.products_df.iloc[similar_indices].copy()
        
        # Add similarity score
        max_dist = max(similar_dists) if similar_dists else 1.0
        if max_dist > 0:
            results['similarity_score'] = 1.0 - (similar_dists / (max_dist * 1.5))
        else:
            results['similarity_score'] = 1.0
        results['similarity_score'] = results['similarity_score'].clip(0.0, 1.0)
        
        return results

    def recommend_for_user(self, user_id, k=10):
        user_id = str(user_id)
        profile_emb = self.get_user_profile_embedding(user_id)
        
        if profile_emb is None:
            # Fallback: Top sustainable products
            print(f"No interactions found for User={user_id}. Falling back to top sustainability products.")
            return self.products_df.sort_values(by='sustainability_index_normalized', ascending=False).head(k).copy()
            
        profile_emb = profile_emb.reshape(1, -1)
        distances, indices = self.faiss_index.search(profile_emb, k)
        
        results = self.products_df.iloc[indices[0]].copy()
        
        # Add personalized match score
        max_dist = max(distances[0]) if len(distances[0]) > 0 else 1.0
        if max_dist > 0:
            results['personalized_match_score'] = 1.0 - (distances[0] / (max_dist * 1.5))
        else:
            results['personalized_match_score'] = 1.0
        results['personalized_match_score'] = results['personalized_match_score'].clip(0.0, 1.0)
        
        return results

    def hybrid_recommend_by_nlq(self, query, user_id=None, k=10, weights=None):
        if weights is None:
            weights = self.default_weights.copy()
            
        user_id = str(user_id) if user_id is not None else None
        
        # If no user profile exists or no user_id is passed, re-normalize weights excluding personalized_ranking
        profile_emb = self.get_user_profile_embedding(user_id) if user_id is not None else None
        if profile_emb is None:
            weights = weights.copy()
            if "personalized_ranking" in weights:
                del weights["personalized_ranking"]
            w_sum = sum(weights.values())
            if w_sum > 0:
                weights = {k: v / w_sum for k, v in weights.items()}
            else:
                weights = {"semantic_similarity": 0.50, "sustainability_score": 0.50}
                
        # --- 1. Semantic Candidates Search (Retrieve 100 products from FAISS) ---
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        query_embedding = query_embedding.astype('float32')
        
        faiss_candidates = min(100, len(self.products_df))
        distances, indices = self.faiss_index.search(query_embedding, faiss_candidates)
        
        candidates = self.products_df.iloc[indices[0]].copy()
        
        # Normalized semantic similarity (0-1 scale)
        max_dist = max(distances[0]) if len(distances[0]) > 0 else 1.0
        candidates['semantic_score'] = 1.0 - (distances[0] / (max_dist * 1.5)) if max_dist > 0 else 1.0
        candidates['semantic_score'] = candidates['semantic_score'].clip(0.0, 1.0)
        
        # --- 2. Calculate Personalized User Similarity (Personalized Ranking) ---
        if profile_emb is not None:
            # Distance from candidate embeddings to user profile embedding
            candidate_indexes = indices[0]
            candidate_embeddings = self.embeddings[candidate_indexes]
            
            # Distance computation: sum of squares
            user_dists = np.sum((candidate_embeddings - profile_emb) ** 2, axis=1)
            max_user_dist = max(user_dists) if len(user_dists) > 0 else 1.0
            candidates['personalized_score'] = 1.0 - (user_dists / (max_user_dist * 1.5)) if max_user_dist > 0 else 1.0
            candidates['personalized_score'] = candidates['personalized_score'].clip(0.0, 1.0)
        else:
            candidates['personalized_score'] = 0.0
            
        # --- 3. Normalization of scores ---
        def min_max_norm(series):
            s_min, s_max = series.min(), series.max()
            if s_max > s_min:
                return (series - s_min) / (s_max - s_min)
            return series.apply(lambda x: 0.5)
            
        candidates['semantic_score_norm'] = min_max_norm(candidates['semantic_score'])
        candidates['sustainability_score_norm'] = candidates['sustainability_index_normalized'] / 100.0
        
        if profile_emb is not None:
            candidates['personalized_score_norm'] = min_max_norm(candidates['personalized_score'])
        else:
            candidates['personalized_score_norm'] = 0.0
            
        # --- 4. Fusion of scores ---
        hybrid_scores = pd.Series(0.0, index=candidates.index)
        if "semantic_similarity" in weights:
            hybrid_scores += candidates['semantic_score_norm'] * weights["semantic_similarity"]
        if "personalized_ranking" in weights and profile_emb is not None:
            hybrid_scores += candidates['personalized_score_norm'] * weights["personalized_ranking"]
        if "sustainability_score" in weights:
            hybrid_scores += candidates['sustainability_score_norm'] * weights["sustainability_score"]
            
        candidates['hybrid_score'] = hybrid_scores
        
        # Sort and return top K
        final_recs = candidates.sort_values(by='hybrid_score', ascending=False).head(k)
        
        # Select key columns to display
        display_cols = [
            'product_id', 'product_name', 'category', 'material', 'packaging_type', 
            'certifications', 'raw_price_usd', 'carbon_footprint_g', 'water_usage_L', 
            'recycled_content_pct', 'eco_score', 'carbon_footprint_score', 'recyclability_score', 
            'packaging_score', 'sustainability_index_normalized', 'hybrid_score', 
            'semantic_score_norm', 'personalized_score_norm', 'sustainability_score_norm'
        ]
        
        # Only keep columns that exist
        valid_cols = [col for col in display_cols if col in final_recs.columns]
        return final_recs[valid_cols]

    def predict_sustainability(self, product_data):
        """
        Expects a dictionary with keys:
        - category, material, packaging_type, certifications, manufacturer_country,
          carbon_footprint_g, water_usage_L, recycled_content_pct, price_usd
        """
        # Create a single-row DataFrame
        df_pred = pd.DataFrame([product_data])
        
        # Ensure all columns needed exist, else fill with default or median
        required_cols = [
            'carbon_footprint_g', 'water_usage_L', 'recycled_content_pct', 'price_usd',
            'category', 'material', 'packaging_type', 'manufacturer_country', 'certifications'
        ]
        for col in required_cols:
            if col not in df_pred.columns:
                df_pred[col] = self.products_df[col].median() if col in ['carbon_footprint_g', 'water_usage_L', 'recycled_content_pct', 'price_usd'] else 'Unknown'
                
        # Calculate Eco, Carbon, Recyclability, and Packaging score formulas to pass to ML model features
        # Note: the models use them as input features too.
        # Let's calculate them using our formulas on the input row
        
        # Eco score formula
        score = 0.0
        mat = df_pred['material'].iloc[0]
        if 'Organic' in mat or 'Bamboo' in mat or 'Recycled' in mat:
            score += 2.0
        elif 'Glass' in mat or 'Linen' in mat or 'Reclaimed' in mat or 'Hemp' in mat:
            score += 1.0
            
        cert = df_pred['certifications'].iloc[0]
        if cert != 'None' and cert != 'Unknown':
            if cert in ['GOTS', 'FSC', 'USDA Organic', 'Fair Trade']:
                score += 1.5
            elif cert in ['GRS', 'OEKO-TEX', 'Energy Star']:
                score += 1.0
                
        # Water usage contribution
        max_water = self.products_df['water_usage_L'].max()
        min_water = self.products_df['water_usage_L'].min()
        if max_water > min_water:
            norm_water = (df_pred['water_usage_L'].iloc[0] - min_water) / (max_water - min_water)
            score -= norm_water * 1.5
        else:
            score -= 0.5
        df_pred['eco_score'] = max(0.0, min(5.0, score))
        
        # Carbon footprint score formula
        max_cf = self.products_df['carbon_footprint_g'].max()
        min_cf = self.products_df['carbon_footprint_g'].min()
        if max_cf > min_cf:
            norm_cf = (df_pred['carbon_footprint_g'].iloc[0] - min_cf) / (max_cf - min_cf)
            df_pred['carbon_footprint_score'] = max(0.0, min(5.0, (1.0 - norm_cf) * 5.0))
        else:
            df_pred['carbon_footprint_score'] = 2.5
            
        # Recyclability score formula
        score_rec = 0.0
        score_rec += (df_pred['recycled_content_pct'].iloc[0] / 100.0) * 2.0
        if 'Recycled' in mat or 'Glass' in mat or 'Aluminum' in mat:
            score_rec += 1.0
        pkg = df_pred['packaging_type'].iloc[0].lower()
        if 'none' in pkg or 'compostable' in pkg:
            score_rec += 1.5
        elif 'recyclable' in pkg or 'glass jar' in pkg:
            score_rec += 1.5
        elif 'paper' in pkg:
            score_rec += 0.5
        df_pred['recyclability_score'] = max(0.0, min(5.0, score_rec))
        
        # Packaging score formula
        score_pkg = 0.0
        if 'none' in pkg or 'compostable' in pkg:
            score_pkg += 5.0
        elif 'recyclable cardboard' in pkg or 'glass jar' in pkg:
            score_pkg += 3.0
        elif 'paper bag' in pkg or 'paper box' in pkg:
            score_pkg += 2.0
        elif 'plastic' in pkg:
            score_pkg += 1.0
        df_pred['packaging_score'] = max(0.0, min(5.0, score_pkg))
        
        # Precomputed index
        df_pred['sustainability_index'] = (
            df_pred['eco_score'] * 0.3 +
            df_pred['carbon_footprint_score'] * 0.3 +
            df_pred['recyclability_score'] * 0.2 +
            df_pred['packaging_score'] * 0.2
        )
        
        max_idx = self.products_df['sustainability_index'].max()
        min_idx = self.products_df['sustainability_index'].min()
        if max_idx > min_idx:
            df_pred['sustainability_index_normalized'] = ((df_pred['sustainability_index'] - min_idx) / (max_idx - min_idx)) * 100.0
        else:
            df_pred['sustainability_index_normalized'] = 50.0
            
        # Scale price_usd to 0-1 (using the bootstrap logic)
        price_max = self.products_df['raw_price_usd'].max()
        price_min = self.products_df['raw_price_usd'].min()
        if price_max > price_min:
            df_pred['price_usd'] = (df_pred['price_usd'] - price_min) / (price_max - price_min)
        else:
            df_pred['price_usd'] = 0.5
            
        # Form features input
        X_in = df_pred[self.feature_columns].copy()
        
        # Apply pipeline transform
        X_proc = self.preprocessor.transform(X_in)
        
        # Run ML inference
        pred_eco = float(self.eco_model.predict(X_proc)[0])
        pred_carbon = float(self.carbon_model.predict(X_proc)[0])
        
        # Ensure predicted values are in range 0-5
        pred_eco = max(0.0, min(5.0, pred_eco))
        pred_carbon = max(0.0, min(5.0, pred_carbon))
        
        # Calculate the resulting predicted sustainability index
        # We replace the engineered eco_score and carbon_footprint_score with the predicted values
        pred_index = (
            pred_eco * 0.3 +
            pred_carbon * 0.3 +
            df_pred['recyclability_score'].iloc[0] * 0.2 +
            df_pred['packaging_score'].iloc[0] * 0.2
        )
        
        if max_idx > min_idx:
            pred_index_normalized = ((pred_index - min_idx) / (max_idx - min_idx)) * 100.0
        else:
            pred_index_normalized = 50.0
            
        return {
            'predicted_eco_score': round(pred_eco, 2),
            'predicted_carbon_footprint_score': round(pred_carbon, 2),
            'recyclability_score': round(float(df_pred['recyclability_score'].iloc[0]), 2),
            'packaging_score': round(float(df_pred['packaging_score'].iloc[0]), 2),
            'predicted_sustainability_index_normalized': round(float(pred_index_normalized), 2)
        }

    def generate_explanation(self, product_row, query=None, user_id=None):
        # Determine highlights of the product
        reasons = []
        
        name = product_row['product_name']
        cat = product_row['category']
        mat = product_row['material']
        pkg = product_row['packaging_type']
        cert = product_row['certifications']
        index_val = product_row['sustainability_index_normalized']
        
        reasons.append(f"**Sustainability Performance**: It scores **{index_val:.1f}/100** on the Sustainability Index, indicating exceptional eco-friendliness.")
        
        # Material evaluation
        if 'Organic' in mat or 'Recycled' in mat or 'Bamboo' in mat or 'Hemp' in mat or 'Linen' in mat:
            reasons.append(f"**Eco-Friendly Material**: Made of *{mat}*, which minimizes resource depletion compared to synthetic alternatives.")
        
        # Packaging evaluation
        if pkg.lower() in ['none', 'compostable bag', 'compostable']:
            reasons.append(f"**Zero-Waste Packaging**: Packaging is *{pkg}*, generating zero plastic waste.")
        elif 'recyclable' in pkg.lower() or 'glass jar' in pkg.lower():
            reasons.append(f"**Circular Packaging**: Uses *{pkg}*, which is 100% recyclable in standard streams.")
            
        # Certifications evaluation
        if cert != 'None' and cert != 'Unknown':
            reasons.append(f"**Verified Standards**: Certified by **{cert}**, confirming strict compliance with environmental/social regulations.")
            
        # Carbon / Water usage comparison
        if product_row['carbon_footprint_g'] < self.products_df['carbon_footprint_g'].median():
            reasons.append(f"**Low Carbon Impact**: Carbon footprint is only {product_row['carbon_footprint_g']}g, which is below the category median.")
            
        if product_row['water_usage_L'] < self.products_df['water_usage_L'].median():
            reasons.append(f"**Water Conservation**: Requires only {product_row['water_usage_L']}L of water during manufacture.")
            
        # Contextual reasons
        if query:
            reasons.append(f"**Search Relevance**: Highly matches your search intent for *\"{query}\"*.")
            
        if user_id:
            profile_emb = self.get_user_profile_embedding(user_id)
            if profile_emb is not None:
                reasons.append(f"**Personal Preferences**: Aligns closely with your personal preference history for sustainable {cat.lower()} products.")
                
        return reasons
