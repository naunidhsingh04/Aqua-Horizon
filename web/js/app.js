// AQUA HORIZON - AI-Driven 7-Day Inundation Forecasting & Disaster Intelligence
let map;
let geojsonLayer;
let allFeatures = [];
let webData = null;
let currentDistrict = null;
let currentDayIndex = 0; // 0 = Today, 1 = Tomorrow, ..., 6 = Day 7
let darkBaseLayer;
let satelliteLayer;
let isSatellite = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    initMap();
    await loadData();
    setupEventListeners();
    startCinematicDescent();
});

// 1. Initialize Map with Earth / Deep Space View
function initMap() {
    map = L.map('map', {
        center: [18.0, 15.0], // Centered out in space / orbital perspective
        zoom: 2,
        minZoom: 2,
        maxZoom: 11,
        zoomControl: false,
        attributionControl: false
    });

    // 100% Free, Zero-API-Key Esri Dark Canvas
    darkBaseLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 16,
        attribution: 'Esri Dark'
    }).addTo(map);

    // 100% Free, Zero-API-Key Esri Satellite Imagery
    satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 18,
        attribution: 'Esri Satellite'
    });
}

// 2. Cinematic Earth-to-India Opening Fly-In (Locks on Whole Country)
function startCinematicDescent() {
    const introEl = document.getElementById('cinematic-intro');
    
    setTimeout(() => {
        // Frame the entire subcontinent of India right at the mid-center
        map.flyToBounds([
            [7.0, 68.0],   // South-West (Kanyakumari, Gujarat coast)
            [36.8, 97.5]   // North-East (Kashmir, Arunachal Pradesh)
        ], {
            duration: 3.0,
            padding: [50, 50],
            easeLinearity: 0.25
        });

        setTimeout(() => {
            if (introEl) {
                introEl.classList.add('hidden');
            }
            if (geojsonLayer) {
                geojsonLayer.eachLayer(layer => {
                    layer.setStyle({ opacity: 0.85, fillOpacity: 0.65 });
                });
            }
        }, 2500);
    }, 700);
}

// 3. Load Datasets (Enriched GeoJSON & Live Meteorological Package)
async function loadData() {
    try {
        const [geoRes, dataRes] = await Promise.all([
            fetch('data/india_districts.geojson'),
            fetch('data/flood_intelligence_data.json')
        ]);

        const geoData = await geoRes.json();
        webData = await dataRes.json();

        renderChoropleth(geoData);
        populateBenchmarkModal();
        populateTicker();
        setupDayChips();

    } catch (err) {
        console.error("Error loading application data:", err);
    }
}

// 4. Color Scale by Risk Probability
function getRiskColor(prob) {
    if (prob >= 0.70) return '#ef4444'; // Critical Red
    if (prob >= 0.45) return '#f97316'; // High Orange
    if (prob >= 0.25) return '#eab308'; // Moderate Yellow
    return '#10b981';                   // Low Green
}

function getRiskCategory(prob) {
    if (prob >= 0.70) return { label: 'Severe Alert', class: 'positive' };
    if (prob >= 0.45) return { label: 'High Risk', class: 'positive' };
    if (prob >= 0.25) return { label: 'Moderate Watch', class: 'warning' };
    return { label: 'Low Risk', class: 'negative' };
}

function getTooltipHTML(props, dayIdx) {
    const daily = props.daily_probs || [0.15];
    const dailyRains = props.daily_rains_mm || [10.0];
    const prob = daily[dayIdx] !== undefined ? daily[dayIdx] : 0.15;
    const rain = dailyRains[dayIdx] !== undefined ? dailyRains[dayIdx] : 10.0;
    const color = getRiskColor(prob);

    return `
        <div class="map-hover-tooltip">
            <div class="tooltip-dist">${props.name || 'District'}</div>
            <div class="tooltip-state">${props.st_nm || 'India'}</div>
            <div class="tooltip-row">
                <span class="tooltip-risk" style="color: ${color};">
                    ${Math.round(prob * 100)}% Risk
                </span>
                <span class="tooltip-rain">🌧️ ${rain} mm</span>
            </div>
        </div>
    `;
}

// 5. Render India Districts Choropleth
function renderChoropleth(geoData) {
    geojsonLayer = L.geoJSON(geoData, {
        style: feature => {
            const props = feature.properties || {};
            const daily = props.daily_probs || [0.15];
            const prob = daily[currentDayIndex] !== undefined ? daily[currentDayIndex] : 0.15;

            return {
                fillColor: getRiskColor(prob),
                weight: 0.8,
                opacity: 0.6,
                color: '#1e293b',
                fillOpacity: 0.65
            };
        },
        onEachFeature: (feature, layer) => {
            allFeatures.push({ feature, layer });

            layer.bindTooltip(getTooltipHTML(feature.properties || {}, currentDayIndex), {
                sticky: true,
                className: 'leaflet-custom-tooltip',
                direction: 'auto',
                offset: [12, -12]
            });

            layer.on({
                mouseover: e => {
                    highlightDistrict(e, feature);
                    if (feature.properties) {
                        currentDistrict = feature.properties;
                        updateDistrictHUD(feature.properties);
                    }
                },
                mouseout: e => resetHighlight(e),
                click: e => selectDistrict(feature, layer)
            });
        }
    }).addTo(map);
}

function highlightDistrict(e, feature) {
    const layer = e.target;
    layer.setStyle({
        weight: 2.2,
        color: '#38bdf8',
        fillOpacity: 0.85
    });
    layer.bringToFront();
}

function resetHighlight(e) {
    geojsonLayer.resetStyle(e.target);
}

function selectDistrict(feature, layer) {
    const props = feature.properties || {};
    currentDistrict = props;

    if (layer && layer.getBounds) {
        map.flyToBounds(layer.getBounds(), { maxZoom: 7.5, duration: 1.2 });
    }

    updateDistrictHUD(props);
}

function selectHotspot(cleanNameSnippet) {
    const target = allFeatures.find(f => {
        const p = f.feature.properties || {};
        return (p.clean_dist && p.clean_dist.includes(cleanNameSnippet)) || 
               (p.name && p.name.toLowerCase().includes(cleanNameSnippet));
    });

    if (target) {
        selectDistrict(target.feature, target.layer);
    }
}

// 6. Update District HUD with Live Meteorological Telemetry (Minimal Pill)
function updateDistrictHUD(props) {
    const nameEl = document.getElementById('hud-district-name');
    const stateEl = document.getElementById('hud-district-state');
    const scoreEl = document.getElementById('hud-prob-score');
    const badgeEl = document.getElementById('hud-prob-badge');
    const probChip = document.getElementById('hud-prob-chip');
    const dfsiEl = document.getElementById('hud-dfsi');
    const liveRainEl = document.getElementById('hud-live-rain');
    const soilMoistureEl = document.getElementById('hud-soil-moisture');
    const pastFloodsEl = document.getElementById('hud-past-floods');
    const quickReasonEl = document.getElementById('hud-quick-reason');

    const dailyProbs = props.daily_probs || [0.2];
    const dailyRains = props.daily_rains_mm || [12.0];
    const prob = dailyProbs[currentDayIndex] !== undefined ? dailyProbs[currentDayIndex] : 0.2;
    const rainMm = dailyRains[currentDayIndex] !== undefined ? dailyRains[currentDayIndex] : 10.0;
    const risk = getRiskCategory(prob);

    nameEl.textContent = props.name || 'Unknown District';
    stateEl.textContent = `${props.st_nm || 'India'} | Zone: ${props.weather_zone || 'Central'}`;
    scoreEl.textContent = `${Math.round(prob * 100)}%`;
    badgeEl.textContent = risk.label.replace(' Alert', '').replace(' Watch', '');

    const color = getRiskColor(prob);
    if (probChip) {
        probChip.style.backgroundColor = `${color}20`;
        probChip.style.borderColor = `${color}55`;
    }
    scoreEl.style.color = color;
    badgeEl.style.color = color;

    // Simple, Plain-English Metrics
    liveRainEl.textContent = `${rainMm} mm`;
    soilMoistureEl.textContent = `${props.soil_moisture_pct || 58}% Wetness`;
    dfsiEl.textContent = `Rank #${props.dfsi_rank || 240} of 640`;
    pastFloodsEl.textContent = `${props.past_5yr_floods || 0} in 5 Years`;

    // 1-Line Simple Explainability
    if (quickReasonEl) {
        const reasons = generateLiveExplainability(props, prob, rainMm);
        quickReasonEl.textContent = reasons[0] || 'Historical water drainage within normal baseline.';
    }
}

function generateLiveExplainability(props, prob, rainMm) {
    const reasons = [];
    if (rainMm >= 15.0) {
        reasons.push(`Heavy incoming rainfall (${rainMm} mm) expected in this river basin.`);
    } else if (rainMm >= 8.0) {
        reasons.push(`Moderate rainfall (${rainMm} mm) forecast for this area.`);
    }

    if (props.soil_moisture_pct >= 75) {
        reasons.push(`Ground is already soaked (${props.soil_moisture_pct}% wet), causing fast surface runoff.`);
    }

    if (props.dfsi_rank && props.dfsi_rank <= 100) {
        reasons.push(`Historically one of India's most flood-vulnerable districts (Rank #${props.dfsi_rank} of 640).`);
    } else if (props.past_5yr_floods >= 2) {
        reasons.push(`Frequent past flood events (${props.past_5yr_floods} recorded in recent 5 years).`);
    } else {
        reasons.push(`Natural river basin drainage fraction is ${props.water_pct}%.`);
    }

    return reasons.slice(0, 3);
}

// 7. 7-Day Forecast Horizon Setup (Single Horizontal Line, Spacious)
function setupDayChips() {
    const dates = webData?.metadata?.forecast_dates || ['Today', '+24h', '+48h', '+72h', 'Day 5', 'Day 6', 'Day 7'];
    const container = document.getElementById('day-chips-container');
    container.innerHTML = '';

    const monthNames = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    dates.forEach((d, idx) => {
        const btn = document.createElement('button');
        btn.className = `day-chip ${idx === 0 ? 'active' : ''}`;
        
        let label = idx === 0 ? 'Today' : idx === 1 ? 'Tomorrow' : `Day ${idx+1}`;
        
        let dateFormatted = '';
        if (d && d.includes('-')) {
            const parts = d.split('-');
            const m = parseInt(parts[1], 10);
            const dayNum = parseInt(parts[2], 10);
            dateFormatted = `${dayNum} ${monthNames[m] || ''}`;
        } else {
            dateFormatted = d;
        }

        btn.innerHTML = `<span class="day-main">${label}</span> <span class="day-sub">(${dateFormatted})</span>`;
        btn.addEventListener('click', () => {
            document.querySelectorAll('.day-chip').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            currentDayIndex = idx;
            recolorDistrictsForDay(idx);
            if (currentDistrict) {
                updateDistrictHUD(currentDistrict);
            }
        });
        container.appendChild(btn);
    });
}

function recolorDistrictsForDay(dayIdx) {
    if (!geojsonLayer) return;

    geojsonLayer.eachLayer(layer => {
        const props = layer.feature.properties || {};
        const daily = props.daily_probs || [0.15];
        const prob = daily[dayIdx] !== undefined ? daily[dayIdx] : 0.15;

        layer.setStyle({
            fillColor: getRiskColor(prob)
        });

        if (layer.setTooltipContent) {
            layer.setTooltipContent(getTooltipHTML(props, dayIdx));
        }
    });
}

// 8. Setup Global Event Listeners
function setupEventListeners() {
    // Replay Space Intro
    const replayBtn = document.getElementById('replay-intro-btn');
    if (replayBtn) {
        replayBtn.addEventListener('click', () => {
            const introEl = document.getElementById('cinematic-intro');
            if (introEl) introEl.classList.remove('hidden');
            map.setView([18.0, 15.0], 2);
            startCinematicDescent();
        });
    }

    // Toggle Satellite View
    const satBtn = document.getElementById('toggle-satellite-btn');
    const satText = document.getElementById('satellite-btn-text');
    if (satBtn) {
        satBtn.addEventListener('click', () => {
            isSatellite = !isSatellite;
            if (isSatellite) {
                map.removeLayer(darkBaseLayer);
                map.addLayer(satelliteLayer);
                satelliteLayer.bringToBack();
                if (satText) satText.textContent = 'Dark Map';
            } else {
                map.removeLayer(satelliteLayer);
                map.addLayer(darkBaseLayer);
                darkBaseLayer.bringToBack();
                if (satText) satText.textContent = 'Satellite';
            }
        });
    }

    // Modals
    document.getElementById('open-benchmark-btn').addEventListener('click', () => {
        document.getElementById('benchmark-modal').classList.add('active');
    });

    document.getElementById('close-benchmark-btn').addEventListener('click', () => {
        document.getElementById('benchmark-modal').classList.remove('active');
    });

    document.getElementById('open-advisory-btn').addEventListener('click', () => {
        generateEmergencyAdvisory();
        document.getElementById('advisory-modal').classList.add('active');
    });

    document.getElementById('close-advisory-btn').addEventListener('click', () => {
        document.getElementById('advisory-modal').classList.remove('active');
    });

    // Copy & WhatsApp Actions
    const copyBtn = document.getElementById('copy-advisory-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const msgEl = document.getElementById('adv-citizen-msg');
            if (!msgEl) return;
            navigator.clipboard.writeText(msgEl.textContent.trim());
            const orig = copyBtn.textContent;
            copyBtn.textContent = '✅ Copied to Clipboard!';
            copyBtn.style.background = 'rgba(34, 197, 94, 0.3)';
            setTimeout(() => {
                copyBtn.textContent = orig;
                copyBtn.style.background = '';
            }, 2000);
        });
    }

    const waBtn = document.getElementById('whatsapp-share-btn');
    if (waBtn) {
        waBtn.addEventListener('click', () => {
            const msgEl = document.getElementById('adv-citizen-msg');
            if (!msgEl) return;
            const msg = msgEl.textContent.trim();
            const waUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent(msg)}`;
            window.open(waUrl, '_blank');
        });
    }
}

// 9. Auto-Drafted Disaster Advisory Generator (Clean, Intuitive, Citizen-Ready)
function generateEmergencyAdvisory() {
    let dist = currentDistrict;
    if (!dist && allFeatures.length > 0) {
        dist = allFeatures[0].feature.properties;
    }
    if (!dist) return;

    const dailyProbs = dist.daily_probs || [0.2];
    const dailyRains = dist.daily_rains_mm || [12.0];
    const prob = dailyProbs[currentDayIndex] !== undefined ? dailyProbs[currentDayIndex] : 0.2;
    const rain = dailyRains[currentDayIndex] !== undefined ? dailyRains[currentDayIndex] : 10.0;
    const probPct = Math.round(prob * 100);
    const risk = getRiskCategory(prob);
    const riskColor = getRiskColor(prob);

    const dateLabel = currentDayIndex === 0 ? 'Today (Live)' : currentDayIndex === 1 ? 'Tomorrow' : `Day ${currentDayIndex + 1}`;

    // Title & Threat Badge
    const titleEl = document.getElementById('adv-district-title');
    const badgeEl = document.getElementById('adv-badge');
    if (titleEl) titleEl.textContent = `${dist.name}, ${dist.st_nm || 'India'}`;
    if (badgeEl) {
        badgeEl.textContent = `${risk.label.toUpperCase()} (${probPct}% RISK)`;
        badgeEl.style.color = riskColor;
        badgeEl.style.borderColor = `${riskColor}60`;
        badgeEl.style.background = `${riskColor}20`;
    }

    // 4 Key Metric Boxes
    const rainEl = document.getElementById('adv-rain');
    const riskEl = document.getElementById('adv-risk');
    const soilEl = document.getElementById('adv-soil');
    const windowEl = document.getElementById('adv-window');
    if (rainEl) rainEl.textContent = `${rain} mm`;
    if (riskEl) riskEl.textContent = `${probPct}%`;
    if (soilEl) soilEl.textContent = `${dist.soil_moisture_pct || 65}%`;
    if (windowEl) windowEl.textContent = dateLabel;

    // Ready-to-Broadcast Citizen Alert (SMS / WhatsApp)
    const citizenMsg = `🚨 AQUA HORIZON FLOOD ALERT: ${dist.name.toUpperCase()} (${dist.st_nm || 'INDIA'})
⚠️ Threat Level: ${risk.label} (${probPct}% Risk)
🌧️ Expected Rain: ${rain} mm | Window: ${dateLabel}
Citizens in low-lying river areas are advised to stay alert and avoid flooded roads and bridges.
For emergency rescue assistance, call 24x7 Helpline: 1077.`;

    const msgEl = document.getElementById('adv-citizen-msg');
    const charCountEl = document.getElementById('adv-char-count');
    if (msgEl) msgEl.textContent = citizenMsg;
    if (charCountEl) charCountEl.textContent = `${citizenMsg.length} characters (SMS / WhatsApp Ready)`;
}

// 10. Populate Ticker & Benchmarks
function populateTicker() {
    const marquee = document.getElementById('ticker-marquee');
    if (!marquee || !webData?.metadata?.ticker_alerts) return;
    
    marquee.innerHTML = webData.metadata.ticker_alerts.map(t => `<span class="ticker-item">${t}</span>`).join(' &nbsp;&bull;&nbsp; ');
}

function populateBenchmarkModal() {
    if (!webData?.metadata?.benchmarks) return;
    const benchmarks = webData.metadata.benchmarks;
    const tbody = document.getElementById('benchmark-tbody');
    tbody.innerHTML = '';

    for (const [modelName, metrics] of Object.entries(benchmarks)) {
        const tr = document.createElement('tr');
        if (modelName.includes('ADASYN')) tr.className = 'champion';
        tr.innerHTML = `
            <td><strong>${modelName}</strong></td>
            <td>${(metrics.Recall * 100).toFixed(1)}%</td>
            <td>${(metrics.Precision * 100).toFixed(1)}%</td>
            <td>${metrics['F1-Score'].toFixed(3)}</td>
            <td>${metrics['PR-AUC'].toFixed(3)}</td>
            <td>${metrics['ROC-AUC'].toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    }
}
