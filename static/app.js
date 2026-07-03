// EcoPulse AI Frontend Application Logic

// --- Application State ---
const state = {
    activeTab: 'dashboard',
    selectedUser: '',
    usersList: [],
    categories: [],
    materials: [],
    certifications: [],
    catalogPage: 1,
    catalogPageSize: 6,
    catalogFilters: {
        category: 'All',
        material: 'All',
        ecoScore: 0.0,
        search: ''
    },
    hybridQuery: '',
    hybridWeights: {
        semantic_similarity: 0.40,
        personalized_ranking: 0.30,
        sustainability_score: 0.30
    }
};

// --- DOM elements cache ---
const el = {
    userSelector: document.getElementById('user-selector'),
    headerUserId: document.getElementById('header-user-id'),
    pageTitle: document.getElementById('page-title'),
    pageSubtitle: document.getElementById('page-subtitle'),
    activeUserBadge: document.getElementById('active-user-badge'),
    
    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    tabs: document.querySelectorAll('.tab-content'),
    
    // Dashboard Stats
    statSustainability: document.getElementById('stat-sustainability'),
    statCarbon: document.getElementById('stat-carbon'),
    statWater: document.getElementById('stat-water'),
    featuredGrid: document.getElementById('featured-products-grid'),
    dashboardChart: document.getElementById('dashboard-chart'),
    dashboardMaterials: document.getElementById('dashboard-materials'),
    btnViewAllFeatured: document.getElementById('btn-view-all-featured'),
    
    // Catalog
    catalogSearch: document.getElementById('catalog-search'),
    catalogFilterCategory: document.getElementById('catalog-filter-category'),
    catalogFilterMaterial: document.getElementById('catalog-filter-material'),
    catalogFilterEco: document.getElementById('catalog-filter-eco'),
    ecoScoreVal: document.getElementById('eco-score-val'),
    catalogProductsGrid: document.getElementById('catalog-products-grid'),
    catalogPrevPage: document.getElementById('catalog-prev-page'),
    catalogNextPage: document.getElementById('catalog-next-page'),
    catalogPageNum: document.getElementById('catalog-page-num'),
    catalogResultsCount: document.getElementById('catalog-results-count'),
    
    // Personalized Portal
    profileUserId: document.getElementById('profile-user-id'),
    profileInterestsDetails: document.getElementById('profile-interests-details'),
    hybridQueryInput: document.getElementById('hybrid-query'),
    btnHybridSearch: document.getElementById('btn-hybrid-search'),
    recommendationsGrid: document.getElementById('recommendations-grid'),
    recommendationsTitle: document.getElementById('recommendations-title'),
    
    // Portal Weights Sliders
    weightSemantic: document.getElementById('weight-semantic'),
    weightPersonal: document.getElementById('weight-personal'),
    weightSustainability: document.getElementById('weight-sustainability'),
    weightSemanticVal: document.getElementById('weight-semantic-val'),
    weightPersonalVal: document.getElementById('weight-personal-val'),
    weightSustainabilityVal: document.getElementById('weight-sustainability-val'),
    weightTotalWarning: document.getElementById('weight-total-warning'),
    
    // Global Weights sliders
    globalWeightSemantic: document.getElementById('global-weight-semantic'),
    globalWeightPersonal: document.getElementById('global-weight-personal'),
    globalWeightSustainability: document.getElementById('global-weight-sustainability'),
    globalWeightSemanticBadge: document.getElementById('global-weight-semantic-badge'),
    globalWeightPersonalBadge: document.getElementById('global-weight-personal-badge'),
    globalWeightSustainabilityBadge: document.getElementById('global-weight-sustainability-badge'),
    btnSaveGlobalWeights: document.getElementById('btn-save-global-weights'),
    visWeightSemantic: document.getElementById('vis-weight-semantic'),
    visWeightPersonal: document.getElementById('vis-weight-personal'),
    visWeightSustainability: document.getElementById('vis-weight-sustainability'),
    
    // Predictor
    predForm: document.getElementById('predictor-form'),
    predCategory: document.getElementById('pred-category'),
    predMaterial: document.getElementById('pred-material'),
    predPackaging: document.getElementById('pred-packaging'),
    predCertifications: document.getElementById('pred-certifications'),
    predCountry: document.getElementById('pred-country'),
    predPrice: document.getElementById('pred-price'),
    predCarbon: document.getElementById('pred-carbon'),
    predCarbonVal: document.getElementById('pred-carbon-val'),
    predWater: document.getElementById('pred-water'),
    predWaterVal: document.getElementById('pred-water-val'),
    predRecycled: document.getElementById('pred-recycled'),
    predRecycledVal: document.getElementById('pred-recycled-val'),
    predictionEmptyState: document.getElementById('prediction-empty-state'),
    predictionResultsCard: document.getElementById('prediction-results-card'),
    predOverallScore: document.getElementById('pred-overall-score'),
    predEcoScore: document.getElementById('pred-eco-score'),
    predCarbonScore: document.getElementById('pred-carbon-score'),
    predRecyclabilityScore: document.getElementById('pred-recyclability-score'),
    predPackagingScore: document.getElementById('pred-packaging-score'),
    predEcoStars: document.getElementById('pred-eco-stars'),
    predCarbonStars: document.getElementById('pred-carbon-stars'),
    predRecyclabilityStars: document.getElementById('pred-recyclability-stars'),
    predPackagingStars: document.getElementById('pred-packaging-stars'),
    predictionReasonsList: document.getElementById('prediction-reasons-list'),
    
    // Modal Details
    explanationModal: document.getElementById('explanation-modal'),
    modalProductName: document.getElementById('modal-product-name'),
    modalCategory: document.getElementById('modal-category'),
    modalSustainabilityIndex: document.getElementById('modal-sustainability-index'),
    modalEcoScore: document.getElementById('modal-eco-score'),
    modalCarbonScore: document.getElementById('modal-carbon-score'),
    modalRecyclabilityScore: document.getElementById('modal-recyclability-score'),
    modalPackagingScore: document.getElementById('modal-packaging-score'),
    modalCarbonRaw: document.getElementById('modal-carbon-raw'),
    modalWaterRaw: document.getElementById('modal-water-raw'),
    modalRecycledRaw: document.getElementById('modal-recycled-raw'),
    modalPackagingRaw: document.getElementById('modal-packaging-raw'),
    modalReasonsList: document.getElementById('modal-reasons-list'),
    modalSimilarProducts: document.getElementById('modal-similar-products'),
    btnCloseModal: document.getElementById('btn-close-modal')
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    await fetchFilterOptions();
    await fetchUsers();
    
    // Trigger initial tab load
    navigate('dashboard');
});

// --- EVENT LISTENERS ---
function setupEventListeners() {
    // SPA Routing Navigation Clicks
    el.navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = item.getAttribute('href').substring(1);
            navigate(tabId);
        });
    });

    // User Selection Profile Changed
    el.userSelector.addEventListener('change', () => {
        state.selectedUser = el.userSelector.value;
        el.headerUserId.textContent = state.selectedUser;
        
        // Refresh items based on active user
        if (state.activeTab === 'dashboard') {
            loadDashboardStats();
        } else if (state.activeTab === 'personalized') {
            updatePersonalizedPortal();
        }
    });

    // Catalog Search/Filter Events
    el.catalogSearch.addEventListener('input', debounce(() => {
        state.catalogFilters.search = el.catalogSearch.value;
        state.catalogPage = 1;
        loadCatalogProducts();
    }, 300));

    el.catalogFilterCategory.addEventListener('change', () => {
        state.catalogFilters.category = el.catalogFilterCategory.value;
        state.catalogPage = 1;
        loadCatalogProducts();
    });

    el.catalogFilterMaterial.addEventListener('change', () => {
        state.catalogFilters.material = el.catalogFilterMaterial.value;
        state.catalogPage = 1;
        loadCatalogProducts();
    });

    el.catalogFilterEco.addEventListener('input', () => {
        const val = parseFloat(el.catalogFilterEco.value);
        el.ecoScoreVal.textContent = val.toFixed(1);
        state.catalogFilters.ecoScore = val;
        state.catalogPage = 1;
        loadCatalogProducts();
    });

    el.catalogPrevPage.addEventListener('click', () => {
        if (state.catalogPage > 1) {
            state.catalogPage--;
            loadCatalogProducts();
        }
    });

    el.catalogNextPage.addEventListener('click', () => {
        state.catalogPage++;
        loadCatalogProducts();
    });

    el.btnViewAllFeatured.addEventListener('click', () => {
        navigate('catalog');
    });

    // Personalized Portal Query Submit
    el.btnHybridSearch.addEventListener('click', () => {
        state.hybridQuery = el.hybridQueryInput.value.trim();
        loadPersonalizedRecommendations(state.hybridQuery);
    });

    el.hybridQueryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            state.hybridQuery = el.hybridQueryInput.value.trim();
            loadPersonalizedRecommendations(state.hybridQuery);
        }
    });

    // Personalized Portal Tag Clicks
    document.querySelectorAll('.suggestion-tag').forEach(tag => {
        tag.addEventListener('click', () => {
            const queryText = tag.textContent;
            el.hybridQueryInput.value = queryText;
            state.hybridQuery = queryText;
            loadPersonalizedRecommendations(queryText);
        });
    });

    // Sync weight sliders in Portal and Global Settings
    setupPortalWeightSync();
    setupGlobalWeightSync();

    // Predictor sliders visual bindings
    el.predCarbon.addEventListener('input', () => el.predCarbonVal.textContent = el.predCarbon.value);
    el.predWater.addEventListener('input', () => el.predWaterVal.textContent = el.predWater.value);
    el.predRecycled.addEventListener('input', () => el.predRecycledVal.textContent = el.predRecycled.value);

    // Predictor Form Submission
    el.predForm.addEventListener('submit', (e) => {
        e.preventDefault();
        runEcoPredictor();
    });

    // Modal Close
    el.btnCloseModal.addEventListener('click', () => {
        el.explanationModal.style.display = 'none';
    });
    
    el.explanationModal.addEventListener('click', (e) => {
        if (e.target === el.explanationModal) {
            el.explanationModal.style.display = 'none';
        }
    });
}

// --- PORTAL WEIGHTS ADJUSTMENT ---
function setupPortalWeightSync() {
    const handleSliderInput = () => {
        const wSemantic = parseFloat(el.weightSemantic.value);
        const wPersonal = parseFloat(el.weightPersonal.value);
        const wSustain = parseFloat(el.weightSustainability.value);
        
        const total = wSemantic + wPersonal + wSustain;
        
        el.weightSemanticVal.textContent = `${Math.round((wSemantic/total)*100)}%`;
        el.weightPersonalVal.textContent = `${Math.round((wPersonal/total)*100)}%`;
        el.weightSustainabilityVal.textContent = `${Math.round((wSustain/total)*100)}%`;
        
        state.hybridWeights.semantic_similarity = wSemantic / total;
        state.hybridWeights.personalized_ranking = wPersonal / total;
        state.hybridWeights.sustainability_score = wSustain / total;
        
        if (total !== 100) {
            el.weightTotalWarning.style.display = 'flex';
        } else {
            el.weightTotalWarning.style.display = 'none';
        }
    };
    
    const handleSliderChange = () => {
        if (state.hybridQuery) {
            loadPersonalizedRecommendations(state.hybridQuery);
        }
    };

    [el.weightSemantic, el.weightPersonal, el.weightSustainability].forEach(slider => {
        slider.addEventListener('input', handleSliderInput);
        slider.addEventListener('change', handleSliderChange);
    });
}

// --- GLOBAL WEIGHTS PANEL SYNC ---
function setupGlobalWeightSync() {
    const handleGlobalSliderInput = () => {
        const wSemantic = parseFloat(el.globalWeightSemantic.value);
        const wPersonal = parseFloat(el.globalWeightPersonal.value);
        const wSustain = parseFloat(el.globalWeightSustainability.value);
        
        const total = wSemantic + wPersonal + wSustain;
        
        const pSemantic = Math.round((wSemantic/total)*100);
        const pPersonal = Math.round((wPersonal/total)*100);
        const pSustain = Math.round((wSustain/total)*100);
        
        el.globalWeightSemanticBadge.textContent = `${pSemantic}%`;
        el.globalWeightPersonalBadge.textContent = `${pPersonal}%`;
        el.globalWeightSustainabilityBadge.textContent = `${pSustain}%`;
        
        el.visWeightSemantic.style.width = `${pSemantic}%`;
        el.visWeightSemantic.textContent = `Semantic (${pSemantic}%)`;
        el.visWeightPersonal.style.width = `${pPersonal}%`;
        el.visWeightPersonal.textContent = `Personalized (${pPersonal}%)`;
        el.visWeightSustainability.style.width = `${pSustain}%`;
        el.visWeightSustainability.textContent = `Sustainability (${pSustain}%)`;
        
        // Sync values to Portal sliders too
        el.weightSemantic.value = pSemantic;
        el.weightPersonal.value = pPersonal;
        el.weightSustainability.value = pSustain;
        
        el.weightSemanticVal.textContent = `${pSemantic}%`;
        el.weightPersonalVal.textContent = `${pPersonal}%`;
        el.weightSustainabilityVal.textContent = `${pSustain}%`;
        
        state.hybridWeights.semantic_similarity = wSemantic / total;
        state.hybridWeights.personalized_ranking = wPersonal / total;
        state.hybridWeights.sustainability_score = wSustain / total;
    };

    [el.globalWeightSemantic, el.globalWeightPersonal, el.globalWeightSustainability].forEach(slider => {
        slider.addEventListener('input', handleGlobalSliderInput);
    });

    el.btnSaveGlobalWeights.addEventListener('click', () => {
        // Just show alert as they are already saved in memory
        alert('Global algorithm weights have been successfully applied across all recommendation vectors!');
        navigate('personalized');
    });
}

// --- NAVIGATION (SPA ROUTING) ---
function navigate(tabId) {
    state.activeTab = tabId;
    
    // Toggle active class on sidebar items
    el.navItems.forEach(item => {
        if (item.getAttribute('href').substring(1) === tabId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    // Toggle active class on pages
    el.tabs.forEach(tab => {
        if (tab.getAttribute('id') === `tab-${tabId}`) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Set page headers
    switch (tabId) {
        case 'dashboard':
            el.pageTitle.textContent = "Dashboard Analytics";
            el.pageSubtitle.textContent = "Real-time carbon and eco-sustainability analytics.";
            loadDashboardStats();
            break;
        case 'catalog':
            el.pageTitle.textContent = "Sustainable Product Catalog";
            el.pageSubtitle.textContent = "Filter and examine materials, recyclability, and carbon offsets.";
            loadCatalogProducts();
            break;
        case 'personalized':
            el.pageTitle.textContent = "EcoPersonal Recommendations";
            el.pageSubtitle.textContent = "Fusing semantic query relevance with your custom interest vector.";
            updatePersonalizedPortal();
            break;
        case 'predictor':
            el.pageTitle.textContent = "Sustainability Predictor Tool";
            el.pageSubtitle.textContent = "Simulate custom materials and certifications to estimate eco index.";
            break;
        case 'weights':
            el.pageTitle.textContent = "Algorithm Hyperparameters";
            el.pageSubtitle.textContent = "Tweak corporate compliance indices and search query parameters.";
            break;
    }
}

// --- DATA FETCHING FUNCTIONS ---

async function fetchUsers() {
    try {
        const response = await fetch('/api/users');
        const data = await response.json();
        state.usersList = data.users;
        
        el.userSelector.innerHTML = '';
        state.usersList.forEach(user => {
            const opt = document.createElement('option');
            opt.value = user;
            opt.textContent = user;
            el.userSelector.appendChild(opt);
        });
        
        if (state.usersList.length > 0) {
            state.selectedUser = state.usersList[0];
            el.userSelector.value = state.selectedUser;
            el.headerUserId.textContent = state.selectedUser;
        }
    } catch (e) {
        console.error("Error loading users database:", e);
    }
}

async function fetchFilterOptions() {
    try {
        const response = await fetch('/api/categories');
        const data = await response.json();
        state.categories = data.categories;
        state.materials = data.materials;
        state.certifications = data.certifications;
        
        // Load Explore Catalog filter selectors
        populateSelect(el.catalogFilterCategory, state.categories, "All Categories");
        populateSelect(el.catalogFilterMaterial, state.materials, "All Materials");
        
        // Load Predictor Form selectors
        populateSelect(el.predCategory, state.categories);
        populateSelect(el.predMaterial, state.materials);
        populateSelect(el.predCertifications, ['None', ...state.certifications]);
    } catch (e) {
        console.error("Error fetching category selections:", e);
    }
}

function populateSelect(selectEl, list, allText = null) {
    selectEl.innerHTML = '';
    if (allText) {
        const optAll = document.createElement('option');
        optAll.value = "All";
        optAll.textContent = allText;
        selectEl.appendChild(optAll);
    }
    list.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        selectEl.appendChild(opt);
    });
}

// --- TAB LOADERS ---

// 1. DASHBOARD LOAD
async function loadDashboardStats() {
    try {
        const response = await fetch(`/api/products?page=1&page_size=3`);
        const data = await response.json();
        
        // Load KPI numbers
        el.statSustainability.textContent = `${data.stats.avg_sustainability_index}/100`;
        el.statCarbon.textContent = `${data.stats.avg_carbon_footprint_g}g`;
        el.statWater.textContent = `${data.stats.avg_water_usage_l}L`;
        
        // Load Featured grid
        // Fetch top 3 products sorted by index (highest first)
        const productsResponse = await fetch(`/api/products?page=1&page_size=3&category=All`);
        const pData = await productsResponse.json();
        
        el.featuredGrid.innerHTML = '';
        pData.products.forEach(p => {
            const card = createProductCard(p);
            el.featuredGrid.appendChild(card);
        });
        
        // Generate Category Insight Mock Chart
        // Call backend API filtered by categories to see stats
        el.dashboardChart.innerHTML = '';
        const cats = state.categories.slice(0, 4); // Take top 4 categories
        
        for (const cat of cats) {
            const catRes = await fetch(`/api/products?page=1&page_size=1&category=${cat}`);
            const catData = await catRes.json();
            const avgCarbon = catData.stats.avg_carbon_footprint_g;
            
            // Generate visual column bar height (max carbon is 6000g, let's normalize bar)
            const heightPct = Math.min(100, Math.max(15, (avgCarbon / 5000) * 100));
            
            const col = document.createElement('div');
            col.className = 'chart-bar-col';
            col.innerHTML = `
                <div class="chart-bar-fill" style="height: ${heightPct}%; background: linear-gradient(to top, var(--primary), var(--water-blue));"></div>
                <div class="chart-bar-label">${cat}<br><b>${Math.round(avgCarbon)}g</b></div>
            `;
            el.dashboardChart.appendChild(col);
        }
        
        // Ingest Materials
        el.dashboardMaterials.innerHTML = '';
        state.materials.slice(0, 8).forEach(mat => {
            const badge = document.createElement('span');
            badge.className = 'material-badge-item';
            badge.innerHTML = `<i class="fa-solid fa-recycle text-primary"></i> ${mat}`;
            el.dashboardMaterials.appendChild(badge);
        });
        
    } catch (e) {
        console.error("Error loading dashboard metrics:", e);
    }
}

// 2. PRODUCT CATALOG LOAD
async function loadCatalogProducts() {
    try {
        const filters = state.catalogFilters;
        const page = state.catalogPage;
        const pageSize = state.catalogPageSize;
        
        let url = `/api/products?page=${page}&page_size=${pageSize}`;
        if (filters.category !== 'All') url += `&category=${encodeURIComponent(filters.category)}`;
        if (filters.material !== 'All') url += `&material=${encodeURIComponent(filters.material)}`;
        if (filters.ecoScore > 0) url += `&min_eco_score=${filters.ecoScore}`;
        if (filters.search) url += `&search=${encodeURIComponent(filters.search)}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        el.catalogResultsCount.textContent = `Showing ${data.total} products`;
        el.catalogPageNum.textContent = `Page ${page} of ${Math.ceil(data.total / pageSize) || 1}`;
        
        // Pagination states
        el.catalogPrevPage.disabled = page === 1;
        el.catalogNextPage.disabled = page * pageSize >= data.total;
        
        // Inject cards
        el.catalogProductsGrid.innerHTML = '';
        if (data.products.length === 0) {
            el.catalogProductsGrid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <i class="fa-solid fa-face-frown"></i>
                    <h4>No products found matching the select filter criteria.</h4>
                </div>
            `;
            return;
        }
        
        data.products.forEach(p => {
            const card = createProductCard(p);
            el.catalogProductsGrid.appendChild(card);
        });
    } catch (e) {
        console.error("Error loading catalog grid:", e);
    }
}

// 3. PERSONALIZED RECOMMENDATIONS PORTAL
async function updatePersonalizedPortal() {
    el.profileUserId.textContent = state.selectedUser;
    
    // Fetch profile metrics (simulate interest from user history)
    try {
        const res = await fetch(`/api/recommend/user?user_id=${state.selectedUser}&limit=20`);
        const data = await res.json();
        
        // Analyze top categories recommended for user
        const catCounts = {};
        data.forEach(p => {
            catCounts[p.category] = (catCounts[p.category] || 0) + 1;
        });
        
        const topCats = Object.keys(catCounts).sort((a, b) => catCounts[b] - catCounts[a]).slice(0, 2);
        const topCatsStr = topCats.length > 0 ? topCats.join(', ') : 'None';
        
        el.profileInterestsDetails.innerHTML = `
            <div class="profile-stat-row">
                <span class="label">Primary Interests</span>
                <span class="value">${topCatsStr}</span>
            </div>
            <div class="profile-stat-row">
                <span class="label">Interactions Logged</span>
                <span class="value">${data.length > 0 ? 'Dynamic Match' : 'New User'}</span>
            </div>
            <div class="profile-stat-row">
                <span class="label">Preferred Materials</span>
                <span class="value">Eco & Recycled</span>
            </div>
        `;
    } catch (e) {
        console.error("Error loading user profile details:", e);
    }
    
    // Load recommendations
    loadPersonalizedRecommendations(state.hybridQuery);
}

async function loadPersonalizedRecommendations(query = "") {
    el.recommendationsGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
            <i class="fa-solid fa-spinner fa-spin"></i>
            <h4>Analyzing vector space and ranking candidate items...</h4>
        </div>
    `;
    
    try {
        let results = [];
        if (!query) {
            // General user personalization
            el.recommendationsTitle.innerHTML = `<i class="fa-solid fa-list-check"></i> Personalized Recommender Feed`;
            const response = await fetch(`/api/recommend/user?user_id=${state.selectedUser}&limit=6`);
            results = await response.json();
        } else {
            // Hybrid search personalization
            el.recommendationsTitle.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Hybrid Match Results for: "${query}"`;
            
            const payload = {
                query: query,
                user_id: state.selectedUser,
                limit: 6,
                weights: {
                    semantic_similarity: state.hybridWeights.semantic_similarity,
                    personalized_ranking: state.hybridWeights.personalized_ranking,
                    sustainability_score: state.hybridWeights.sustainability_score
                }
            };
            
            const response = await fetch('/api/recommend/hybrid', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            results = await response.json();
        }
        
        el.recommendationsGrid.innerHTML = '';
        if (results.length === 0) {
            el.recommendationsGrid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <i class="fa-solid fa-face-frown"></i>
                    <h4>No personalized candidates matched this vector lookup. Try another search terms.</h4>
                </div>
            `;
            return;
        }
        
        results.forEach(p => {
            // In hybrid searches, show score details
            const card = createProductCard(p, true);
            el.recommendationsGrid.appendChild(card);
        });
        
    } catch (e) {
        console.error("Error loading recommendations:", e);
    }
}

// 4. RUN ECO SUSTAINABILITY PREDICTOR
async function runEcoPredictor() {
    el.predictionEmptyState.style.display = 'none';
    el.predictionResultsCard.style.display = 'none';
    
    // Fetch input values
    const payload = {
        category: el.predCategory.value,
        material: el.predMaterial.value,
        packaging_type: el.predPackaging.value,
        certifications: el.predCertifications.value,
        manufacturer_country: el.predCountry.value,
        carbon_footprint_g: parseInt(el.predCarbon.value),
        water_usage_L: parseInt(el.predWater.value),
        recycled_content_pct: parseInt(el.predRecycled.value),
        price_usd: parseFloat(el.predPrice.value)
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        
        // Populate predicted overall values
        el.predOverallScore.textContent = `${data.predicted_sustainability_index_normalized.toFixed(1)}/100`;
        el.predEcoScore.textContent = `${data.predicted_eco_score.toFixed(1)}/5`;
        el.predCarbonScore.textContent = `${data.predicted_carbon_footprint_score.toFixed(1)}/5`;
        el.predRecyclabilityScore.textContent = `${data.recyclability_score.toFixed(1)}/5`;
        el.predPackagingScore.textContent = `${data.packaging_score.toFixed(1)}/5`;
        
        // Populate star ratings
        injectStars(el.predEcoStars, data.predicted_eco_score);
        injectStars(el.predCarbonStars, data.predicted_carbon_footprint_score);
        injectStars(el.predRecyclabilityStars, data.recyclability_score);
        injectStars(el.predPackagingStars, data.packaging_score);
        
        // Generate insights text dynamically
        el.predictionReasonsList.innerHTML = '';
        const index = data.predicted_sustainability_index_normalized;
        
        let insightLevel = "Standard Eco product";
        if (index >= 85) insightLevel = "Outstanding sustainability profile (Circular Gold Class)";
        else if (index >= 70) insightLevel = "Highly eco-conscious material combination";
        else if (index < 45) insightLevel = "Suboptimal packaging or carbon footprint metrics detected";
        
        const reasons = [
            `Overall index: **${insightLevel}**.`,
            `The predicted Eco Score is **${data.predicted_eco_score.toFixed(1)}/5**, reflecting the raw material and certification weightings.`,
            `Predicted Carbon Credit rating: **${data.predicted_carbon_footprint_score.toFixed(1)}/5**, which evaluates manufacturing greenhouse impacts.`,
            `Packaging recyclability scores **${data.packaging_score.toFixed(1)}/5** based on compostability guidelines.`
        ];
        
        reasons.forEach(r => {
            const li = document.createElement('li');
            li.innerHTML = r.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            el.predictionReasonsList.appendChild(li);
        });
        
        el.predictionResultsCard.style.display = 'block';
    } catch (e) {
        console.error("Error performing eco prediction:", e);
    }
}

// --- VIEW DETAILS MODAL ---
async function viewProductDetails(p) {
    el.modalProductName.textContent = p.product_name;
    el.modalCategory.textContent = p.category;
    el.modalSustainabilityIndex.textContent = p.sustainability_index_normalized.toFixed(1);
    el.modalEcoScore.textContent = `${p.eco_score.toFixed(1)}/5`;
    el.modalCarbonScore.textContent = `${p.carbon_footprint_score.toFixed(1)}/5`;
    el.modalRecyclabilityScore.textContent = `${p.recyclability_score.toFixed(1)}/5`;
    el.modalPackagingScore.textContent = `${p.packaging_score.toFixed(1)}/5`;
    
    el.modalCarbonRaw.textContent = `${p.carbon_footprint_g} grams CO2e`;
    el.modalWaterRaw.textContent = `${p.water_usage_L} Liters`;
    el.modalRecycledRaw.textContent = `${p.recycled_content_pct}% recycled content`;
    el.modalPackagingRaw.textContent = p.packaging_type;
    
    // Log view interaction in the background to update profile vector
    logViewInteraction(p.product_id);
    
    // Show modal loading indicators
    el.modalReasonsList.innerHTML = '<li>Loading explanation nodes...</li>';
    el.modalSimilarProducts.innerHTML = '<div>Loading similar products...</div>';
    el.explanationModal.style.display = 'flex';
    
    // Fetch reasons explanation
    try {
        let explainUrl = `/api/explain?product_id=${p.product_id}`;
        if (state.hybridQuery) explainUrl += `&query=${encodeURIComponent(state.hybridQuery)}`;
        if (state.selectedUser) explainUrl += `&user_id=${state.selectedUser}`;
        
        const explainRes = await fetch(explainUrl);
        const explainData = await explainRes.json();
        
        el.modalReasonsList.innerHTML = '';
        explainData.reasons.forEach(r => {
            const li = document.createElement('li');
            // Bold parser
            li.innerHTML = r.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            el.modalReasonsList.appendChild(li);
        });
    } catch (e) {
        console.error("Error explaining product:", e);
    }
    
    // Fetch similar products
    try {
        const similarRes = await fetch(`/api/recommend/similar?product_id=${p.product_id}&limit=3`);
        const similarData = await similarRes.json();
        
        el.modalSimilarProducts.innerHTML = '';
        if (similarData.length === 0) {
            el.modalSimilarProducts.innerHTML = '<div class="empty-state-text">No similar products cataloged.</div>';
            return;
        }
        
        similarData.forEach(sp => {
            const smCard = document.createElement('div');
            smCard.className = 'similar-product-small-card';
            smCard.innerHTML = `
                <h6>${sp.product_name}</h6>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:auto;">
                    <span class="small-score"><i class="fa-solid fa-leaf"></i> ${sp.sustainability_index_normalized.toFixed(0)}</span>
                    <span class="small-price">$${sp.raw_price_usd.toFixed(2)}</span>
                </div>
            `;
            smCard.addEventListener('click', () => {
                viewProductDetails(sp);
            });
            el.modalSimilarProducts.appendChild(smCard);
        });
    } catch (e) {
        console.error("Error fetching similar products:", e);
    }
}

// Log a click interaction to the database
async function logViewInteraction(productId) {
    if (!state.selectedUser) return;
    try {
        await fetch('/api/interact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: state.selectedUser,
                product_id: productId,
                interaction_score: 3 // Score 3 represents a standard item click/view
            })
        });
    } catch (e) {
        console.error("Error logging interaction:", e);
    }
}

// --- CARD CREATION HELPER ---
function createProductCard(p, isRecommendation = false) {
    const div = document.createElement('div');
    div.className = 'product-card';
    
    // Check if score exists to display (personal or hybrid)
    let scoreBadge = '';
    if (isRecommendation && p.hybrid_score !== undefined) {
        scoreBadge = `<span class="eco-rating-badge"><i class="fa-solid fa-chart-line"></i> Match: ${(p.hybrid_score * 100).toFixed(0)}%</span>`;
    } else {
        scoreBadge = `<span class="eco-rating-badge"><i class="fa-solid fa-leaf"></i> Eco Score: ${p.eco_score.toFixed(1)}</span>`;
    }
    
    div.innerHTML = `
        <div class="card-header">
            <span class="category-tag">${p.category}</span>
            ${scoreBadge}
        </div>
        <h4 class="product-title">${p.product_name}</h4>
        
        <div class="sustainability-index-bar">
            <div class="bar-header">
                <span>Sustainability Index</span>
                <span class="score-val">${p.sustainability_index_normalized.toFixed(1)}/100</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${p.sustainability_index_normalized}%"></div>
            </div>
        </div>

        <div class="card-indicators">
            <div class="indicator-item carbon">
                <i class="fa-solid fa-cloud-arrow-down"></i>
                <div>
                    Carbon: <span>${p.carbon_footprint_g}g</span>
                </div>
            </div>
            <div class="indicator-item water">
                <i class="fa-solid fa-droplet"></i>
                <div>
                    Water: <span>${p.water_usage_L}L</span>
                </div>
            </div>
        </div>
        
        <div class="card-footer">
            <span class="product-price">$${p.raw_price_usd.toFixed(2)}</span>
            <button class="btn btn-secondary text-btn btn-details"><i class="fa-solid fa-circle-info"></i> Examine</button>
        </div>
    `;
    
    // Bind click events
    div.querySelector('.btn-details').addEventListener('click', (e) => {
        e.stopPropagation();
        viewProductDetails(p);
    });
    
    div.addEventListener('click', () => {
        // Record click to dynamically adapt profile
        logViewInteraction(p.product_id);
        viewProductDetails(p);
    });
    
    return div;
}

// Star rating injector helper
function injectStars(container, score) {
    container.innerHTML = '';
    const rounded = Math.round(score);
    for (let i = 1; i <= 5; i++) {
        const star = document.createElement('i');
        star.className = i <= rounded ? 'fa-solid fa-star' : 'fa-regular fa-star';
        container.appendChild(star);
    }
}

// Debounce helper for search inputs
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
