import os
import numpy as np
import pandas as pd
import joblib
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def run_bootstrap():
    print("--- STEP 1: Generating Synthetic Products ---")
    np.random.seed(42)
    
    categories = ['Apparel', 'Home Goods', 'Electronics', 'Food', 'Personal Care', 'Footwear']
    
    materials_by_cat = {
        'Apparel': ['Organic Cotton', 'Recycled Polyester', 'Linen', 'Hemp', 'Tencel', 'Organic Wool'],
        'Home Goods': ['Bamboo', 'Glass', 'Reclaimed Wood', 'Cork', 'Clay', 'Recycled Aluminum'],
        'Electronics': ['Recycled Plastic', 'Bio-Plastic', 'Aluminum', 'Recycled Copper', 'Bamboo Fiber'],
        'Food': ['Organic Oats', 'Local Fruits', 'Fair Trade Coffee', 'Organic Cocoa', 'Ancient Grains'],
        'Personal Care': ['Bamboo Fiber', 'Organic Aloe', 'Coconut Oil', 'Shea Butter', 'Organic Herbs'],
        'Footwear': ['Recycled Rubber', 'Organic Canvas', 'Vegan Leather', 'Cork', 'Natural Latex']
    }
    
    packaging_types = ['None', 'Compostable Bag', 'Recyclable Cardboard', 'Glass Jar', 'Paper Box', 'Plastic Bag', 'Plastic Film']
    
    certifications_by_cat = {
        'Apparel': ['GOTS', 'GRS', 'OEKO-TEX', 'Fair Trade', None],
        'Home Goods': ['FSC', 'Fair Trade', None],
        'Electronics': ['Energy Star', 'RoHS', None],
        'Food': ['USDA Organic', 'Fair Trade', None],
        'Personal Care': ['USDA Organic', 'Leaping Bunny', 'Fair Trade', None],
        'Footwear': ['GRS', 'Fair Trade', None]
    }
    
    countries = ['USA', 'Germany', 'Portugal', 'Brazil', 'India', 'China', 'Vietnam', 'Japan', 'France', 'Kenya']
    
    product_names_by_cat = {
        'Apparel': ['T-Shirt', 'Jeans', 'Socks', 'Jacket', 'Dress', 'Sweater', 'Activewear Pants', 'Scarf'],
        'Home Goods': ['Reusable Straws', 'Bento Box', 'Cutting Board', 'Bed Sheets', 'Coffee Mug', 'Dinner Set', 'Bath Towel'],
        'Electronics': ['Solar Charger', 'Eco Phone Case', 'Bamboo Keyboard', 'Recycled Cables', 'LED Desk Lamp', 'Solar Powerbank'],
        'Food': ['Granola Bar', 'Dried Fruit Mix', 'Coffee Beans', 'Dark Chocolate', 'Quinoa Pack', 'Herbal Tea Bag'],
        'Personal Care': ['Bamboo Toothbrush', 'Solid Shampoo Bar', 'Organic Soap', 'Reusable Makeup Pads', 'Lip Balm'],
        'Footwear': ['Running Shoes', 'Casual Sneakers', 'Slippers', 'Sandals', 'Boots']
    }

    products = []
    num_products = 500
    
    for i in range(1, num_products + 1):
        cat = np.random.choice(categories)
        material = np.random.choice(materials_by_cat[cat])
        packaging = np.random.choice(packaging_types)
        
        certs = certifications_by_cat[cat]
        cert = np.random.choice(certs)
        
        country = np.random.choice(countries)
        item_type = np.random.choice(product_names_by_cat[cat])
        
        name = f"{material} {item_type}"
        
        # Base distributions for footprint and water usage
        if cat == 'Electronics':
            carbon_footprint = np.random.randint(1500, 5000)
            water_usage = np.random.randint(500, 3000)
            price = round(np.random.uniform(25.0, 150.0), 2)
        elif cat == 'Apparel' or cat == 'Footwear':
            carbon_footprint = np.random.randint(800, 2500)
            water_usage = np.random.randint(1000, 6000)
            price = round(np.random.uniform(15.0, 120.0), 2)
        elif cat == 'Home Goods':
            carbon_footprint = np.random.randint(200, 1500)
            water_usage = np.random.randint(50, 1000)
            price = round(np.random.uniform(10.0, 80.0), 2)
        else: # Food or Personal Care
            carbon_footprint = np.random.randint(50, 600)
            water_usage = np.random.randint(10, 500)
            price = round(np.random.uniform(3.0, 30.0), 2)
            
        recycled_content = int(np.random.choice([0, 20, 50, 80, 95, 100], p=[0.3, 0.1, 0.1, 0.2, 0.2, 0.1]))
        if 'Recycled' in material:
            recycled_content = int(np.random.randint(80, 101))
            
        products.append({
            'product_id': str(i),
            'product_name': name,
            'category': cat,
            'material': material,
            'carbon_footprint_g': carbon_footprint,
            'water_usage_L': water_usage,
            'recycled_content_pct': recycled_content,
            'packaging_type': packaging,
            'certifications': cert if cert is not None else 'None',
            'price_usd': price,
            'manufacturer_country': country
        })
        
    df = pd.DataFrame(products)
    
    print("--- STEP 2: Applying Feature Engineering Rules ---")
    
    # 2.1 Calculate Eco Score (0 to 5)
    def calculate_eco_score(row):
        score = 0.0
        mat = row['material']
        if 'Organic' in mat or 'Bamboo' in mat or 'Recycled' in mat:
            score += 2.0
        elif 'Glass' in mat or 'Linen' in mat or 'Reclaimed' in mat or 'Hemp' in mat:
            score += 1.0
            
        cert = row['certifications']
        if cert != 'None':
            if cert in ['GOTS', 'FSC', 'USDA Organic', 'Fair Trade']:
                score += 1.5
            elif cert in ['GRS', 'OEKO-TEX', 'Energy Star']:
                score += 1.0
                
        # Water usage contribution (lower water usage = higher score)
        max_water = df['water_usage_L'].max()
        min_water = df['water_usage_L'].min()
        if max_water > min_water:
            norm_water = (row['water_usage_L'] - min_water) / (max_water - min_water)
            score -= norm_water * 1.5
        else:
            score -= 0.5
            
        return max(0.0, min(5.0, score))

    df['eco_score'] = df.apply(calculate_eco_score, axis=1)
    
    # 2.2 Calculate Carbon Footprint Score (0 to 5)
    def calculate_carbon_footprint_score(row):
        max_cf = df['carbon_footprint_g'].max()
        min_cf = df['carbon_footprint_g'].min()
        if max_cf == min_cf:
            return 2.5
        norm_cf = (row['carbon_footprint_g'] - min_cf) / (max_cf - min_cf)
        score = (1.0 - norm_cf) * 5.0
        return max(0.0, min(5.0, score))
        
    df['carbon_footprint_score'] = df.apply(calculate_carbon_footprint_score, axis=1)
    
    # 2.3 Calculate Recyclability Score (0 to 5)
    def calculate_recyclability_score(row):
        score = 0.0
        score += (row['recycled_content_pct'] / 100.0) * 2.0
        if 'Recycled' in row['material'] or 'Glass' in row['material'] or 'Aluminum' in row['material']:
            score += 1.0
        pkg = row['packaging_type'].lower()
        if 'none' in pkg or 'compostable' in pkg:
            score += 1.5
        elif 'recyclable' in pkg or 'glass jar' in pkg:
            score += 1.5
        elif 'paper' in pkg:
            score += 0.5
        return max(0.0, min(5.0, score))
        
    df['recyclability_score'] = df.apply(calculate_recyclability_score, axis=1)
    
    # 2.4 Calculate Packaging Score (0 to 5)
    def calculate_packaging_score(row):
        score = 0.0
        pkg = row['packaging_type'].lower()
        if 'none' in pkg or 'compostable' in pkg:
            score += 5.0
        elif 'recyclable cardboard' in pkg or 'glass jar' in pkg:
            score += 3.0
        elif 'paper bag' in pkg or 'paper box' in pkg:
            score += 2.0
        elif 'plastic' in pkg:
            score += 1.0
        return max(0.0, min(5.0, score))
        
    df['packaging_score'] = df.apply(calculate_packaging_score, axis=1)
    
    # 2.5 Combine into Sustainability Index
    df['sustainability_index'] = (
        df['eco_score'] * 0.3 +
        df['carbon_footprint_score'] * 0.3 +
        df['recyclability_score'] * 0.2 +
        df['packaging_score'] * 0.2
    )
    
    max_idx = df['sustainability_index'].max()
    min_idx = df['sustainability_index'].min()
    if max_idx > min_idx:
        df['sustainability_index_normalized'] = ((df['sustainability_index'] - min_idx) / (max_idx - min_idx)) * 100.0
    else:
        df['sustainability_index_normalized'] = 50.0
        
    # Scale price_usd to 0-1 for the model pipeline (retaining raw_price for display)
    df['raw_price_usd'] = df['price_usd']
    price_max = df['price_usd'].max()
    price_min = df['price_usd'].min()
    if price_max > price_min:
        df['price_usd'] = (df['price_usd'] - price_min) / (price_max - price_min)
    else:
        df['price_usd'] = 0.5
        
    # 2.6 Generate Textual Descriptions for SentenceTransformer
    def generate_product_description(row):
        parts = []
        if pd.notna(row['product_name']): parts.append(f"Product Name: {row['product_name']}.")
        if pd.notna(row['category']): parts.append(f"Category: {row['category']}.")
        if pd.notna(row['material']): parts.append(f"Material: {row['material']}.")
        if pd.notna(row['packaging_type']): parts.append(f"Packaging: {row['packaging_type']}.")
        if pd.notna(row['manufacturer_country']): parts.append(f"Manufactured in: {row['manufacturer_country']}.")
        if pd.notna(row['eco_score']): parts.append(f"Eco Score: {row['eco_score']:.1f}/5.")
        if pd.notna(row['carbon_footprint_score']): parts.append(f"Carbon Footprint Score: {row['carbon_footprint_score']:.1f}/5.")
        if pd.notna(row['recyclability_score']): parts.append(f"Recyclability Score: {row['recyclability_score']:.1f}/5.")
        if pd.notna(row['packaging_score']): parts.append(f"Packaging Score: {row['packaging_score']:.1f}/5.")
        if pd.notna(row['sustainability_index_normalized']): parts.append(f"Overall Sustainability Index: {row['sustainability_index_normalized']:.1f}/100.")
        return " ".join(parts)

    df['product_description'] = df.apply(generate_product_description, axis=1)
    
    # Save the processed dataset
    df.to_csv('processed_products.csv', index=False)
    print("Saved processed products to 'processed_products.csv'")
    
    print("--- STEP 3: Generating Product Embeddings using Sentence-BERT ---")
    print("Loading SentenceTransformer('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = df['product_description'].tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True, convert_to_numpy=True)
    
    np.save('product_embeddings.npy', embeddings)
    print("Saved product embeddings to 'product_embeddings.npy'")
    
    print("--- STEP 4: Building and Saving FAISS Vector Index ---")
    embedding_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embeddings.astype('float32'))
    faiss.write_index(faiss_index, 'faiss_product_index.bin')
    print("Saved FAISS index to 'faiss_product_index.bin'")
    
    print("--- STEP 5: Fitting ML Preprocessing Pipeline and Training Predictor Models ---")
    feature_columns = [
        'carbon_footprint_g',
        'water_usage_L',
        'recycled_content_pct',
        'price_usd',
        'recyclability_score',
        'packaging_score',
        'sustainability_index',
        'sustainability_index_normalized',
        'category',
        'material',
        'packaging_type',
        'manufacturer_country',
        'certifications'
    ]
    
    numerical_features = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"Numerical Features: {numerical_features}")
    print(f"Categorical Features: {categorical_features}")
    
    X = df[feature_columns].copy()
    Y_eco = df['eco_score'].copy()
    Y_carbon = df['carbon_footprint_score'].copy()
    
    # Preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Train random forest regressors as replacements for the DNNs
    eco_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
    eco_predictor.fit(X_processed, Y_eco)
    
    carbon_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
    carbon_predictor.fit(X_processed, Y_carbon)
    
    # Save the artifacts
    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    joblib.dump(eco_predictor, 'eco_model.joblib')
    joblib.dump(carbon_predictor, 'carbon_credit_model.joblib')
    print("Fitted and saved preprocessor, features, and model regressors.")
    
    print("--- STEP 6: Generating Synthetic Interaction Data ---")
    num_users = 100
    num_interactions = 3000
    
    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    product_ids = df['product_id'].tolist()
    
    sampled_users = np.random.choice(user_ids, size=num_interactions).tolist()
    sampled_products = np.random.choice(product_ids, size=num_interactions).tolist()
    scores = np.random.randint(1, 6, size=num_interactions).tolist()
    
    interactions_df = pd.DataFrame({
        'user_id': sampled_users,
        'product_id': sampled_products,
        'interaction_score': scores
    })
    
    # Add high scores for user_1 to user_5 to specific categories to mock distinct user interests
    for idx, row in interactions_df.iterrows():
        uid = row['user_id']
        pid = row['product_id']
        p_row = df[df['product_id'] == pid].iloc[0]
        
        # User 1 likes Apparel
        if uid == 'user_1' and p_row['category'] == 'Apparel':
            interactions_df.at[idx, 'interaction_score'] = 5
        # User 2 likes Electronics
        elif uid == 'user_2' and p_row['category'] == 'Electronics':
            interactions_df.at[idx, 'interaction_score'] = 5
        # User 3 likes Home Goods
        elif uid == 'user_3' and p_row['category'] == 'Home Goods':
            interactions_df.at[idx, 'interaction_score'] = 5
            
    interactions_df.to_csv('interactions.csv', index=False)
    print("Saved user interactions to 'interactions.csv'")
    print("--- BOOTSTRAP COMPLETE! ---")

if __name__ == '__main__':
    run_bootstrap()
