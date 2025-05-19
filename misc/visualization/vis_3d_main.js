const container = document.getElementById('deck-container');
const slider = document.getElementById('pointCloudSlider');
const animationSpeedInput = document.getElementById('animationSpeed');
const menuButton = document.getElementById('menu-button');
const toggleMetricButton = document.getElementById('toggle-metric-button');
const toggleConfigButton = document.getElementById('toggle-config-button');
const menuContainer = document.getElementById('menu-container');
const bubbleSizeInput = document.getElementById('bubbleSize');
const pathColorInput = document.getElementById('pathColor');
const pathSizeInput = document.getElementById('pathSize');
const metricContainer = document.getElementById('metric-container');
const configContainer = document.getElementById('config-container');
const runsDropdown = document.getElementById('runs-dropdown');
const sweepsDropdown = document.getElementById('sweeps-dropdown');

// State object to hold shared variables
const appState = {
  isPlaying: false,
  playInterval: null,
  metricData: [],
  colorsPerTimestep: [],
  updateLayers: null,
  deckGL: null,
  runConfig: null,
  zipContent: null, // Add zipContent to hold extracted files
  pointClouds: [], // Add pointClouds to hold point cloud data
  currentViewState: null // Track current view state for transitions
};

// Define handleKeyPress
function handleKeyPress(event) {
  if (event.key === ' ' || event.code === 'Space') {
    event.preventDefault();
    togglePlayStop();
  } else if ((event.key === '+' || event.key === '=') && !appState.isPlaying) {
    event.preventDefault();
    nextFrame();
  } else if (event.key === '-' && !appState.isPlaying) {
    event.preventDefault();
    previousFrame();
  } else if (event.key === 'c' || event.key === 'C') {
    event.preventDefault();
    centerOnCurrentPath();
  }
}

// Add event listener once
document.addEventListener('keydown', handleKeyPress);

// Function to center the map on the current path position
function centerOnCurrentPath() {
  const currentIndex = parseInt(slider.value);
  const pathPoints = appState.paths[currentIndex];
  
  if (pathPoints && pathPoints.length > 0) {
    // Get the last position in the current path
    const lastPosition = pathPoints[pathPoints.length - 1];
    
    // Create a new initialViewState with the desired target position
    const newViewState = {
      longitude: lastPosition[0],
      latitude: lastPosition[1],
      zoom: appState.currentViewState ? appState.currentViewState.zoom : 11.5,
      pitch: appState.currentViewState ? appState.currentViewState.pitch : 0,
      bearing: appState.currentViewState ? appState.currentViewState.bearing : 0,
      transitionDuration: 100,
      transitionInterpolator: new deck.FlyToInterpolator()
    };
    
    // Use setProps with initialViewState to maintain controller functionality
    appState.deckGL.setProps({
      initialViewState: newViewState
    });
  }
}

function togglePlayStop() {
  if (appState.isPlaying) {
    clearInterval(appState.playInterval);
  } else {
    const speed = parseInt(animationSpeedInput.value);
    appState.playInterval = setInterval(() => {
      let currentIndex = parseInt(slider.value);
      currentIndex = (currentIndex + 1) % appState.colorsPerTimestep.length;
      slider.value = currentIndex;
      appState.updateLayers(currentIndex);
    }, speed);
  }
  appState.isPlaying = !appState.isPlaying;
}

function nextFrame() {
  let currentIndex = parseInt(slider.value);
  if (currentIndex < appState.colorsPerTimestep.length - 1) {
    currentIndex += 1;
    slider.value = currentIndex;
    appState.updateLayers(currentIndex);
  }
}

function previousFrame() {
  let currentIndex = parseInt(slider.value);
  if (currentIndex > 0) {
    currentIndex -= 1;
    slider.value = currentIndex;
    appState.updateLayers(currentIndex);
  }
}

menuButton.addEventListener('click', () => {
  menuContainer.style.display = menuContainer.style.display === 'none' ? 'block' : 'none';
});

toggleMetricButton.addEventListener('click', () => {
  metricContainer.style.display = metricContainer.style.display === 'none' ? 'block' : 'none';
});

toggleConfigButton.addEventListener('click', () => {
  configContainer.style.display = configContainer.style.display === 'none' ? 'block' : 'none';
});

// Load and extract the zip file
function loadZipFile() {
  fetch('html_data.zip')
    .then(response => {
      if (response.ok) return response.arrayBuffer();
      else throw new Error('Network response was not ok.');
    })
    .then(data => {
      return JSZip.loadAsync(data);
    })
    .then(zip => {
      appState.zipContent = zip;
      // Now proceed to load sweep IDs and set up event listener
      loadSweepIds();
    })
    .catch(error => {
      console.error('Error loading zip file:', error);
    });
}

// Call loadZipFile to start the process
loadZipFile();

function loadSweepIds() {
  appState.zipContent
    .file('sweep_ids.json')
    .async('string')
    .then(text => {
      const sweepIds = JSON.parse(text);
      sweepIds.forEach(sweepId => {
        const option = document.createElement('option');
        option.value = sweepId;
        option.text = sweepId;
        sweepsDropdown.appendChild(option);
      });

      // Load run IDs for the first sweep_id by default
      loadRunIds(sweepIds[0]);
    })
    .catch(error => {
      console.error('Error loading sweep_ids:', error);
    });
}

function loadRunIds(sweepId) {
  // Clear the runs-dropdown
  runsDropdown.innerHTML = '';

  appState.zipContent
    .file(`${sweepId}/run_ids.json`)
    .async('string')
    .then(text => {
      const runIds = JSON.parse(text);
      runIds.forEach(runId => {
        const option = document.createElement('option');
        option.value = runId;
        option.text = runId;
        runsDropdown.appendChild(option);
      });

      // Load data for the first run_id by default
      loadData(sweepId, runIds[0]);
    })
    .catch(error => {
      console.error('Error loading run_ids:', error);
    });
}

// Add event listeners
sweepsDropdown.addEventListener('change', event => {
  const sweepId = event.target.value;
  loadRunIds(sweepId);
});

runsDropdown.addEventListener('change', event => {
  const sweepId = sweepsDropdown.value;
  const runId = event.target.value;
  loadData(sweepId, runId);
});

function loadData(sweepId, runId) {
  // Pause the animation and reset the slider
  if (appState.isPlaying) {
    clearInterval(appState.playInterval);
    appState.isPlaying = false;
  }
  slider.value = 0;

  appState.zipContent
    .file(`${sweepId}/${runId}/center_coordinates.json`)
    .async('string')
    .then(text => {
      const centerData = JSON.parse(text);
      const centerLongitude = centerData.center_lon;
      const centerLatitude = centerData.center_lat;

      const initialViewState = {
        longitude: centerLongitude,
        latitude: centerLatitude,
        zoom: 11.5,
        pitch: 0,
        bearing: 0
      };

      const terrainLayer = new deck.TerrainLayer({
        id: 'terrain-layer',
        elevationData: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
        texture:
          'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        elevationDecoder: {
          rScaler: 256,
          gScaler: 1,
          bScaler: 1 / 256,
          offset: -32768
        },
        maxZoom: 15,
        operation: 'terrain+draw'
      });

      if (!appState.deckGL) {
        appState.deckGL = new deck.DeckGL({
          container,
          initialViewState,
          controller: {
            type: deck.MapController,
            maxPitch: 85
          },
          layers: [terrainLayer],
          onViewStateChange: ({viewState}) => {
            appState.currentViewState = viewState;
          }
        });
        // Initialize currentViewState with the initialViewState
        appState.currentViewState = initialViewState;
      } else {
        appState.deckGL.setProps({
          initialViewState,
          layers: [terrainLayer],
          onViewStateChange: ({viewState}) => {
            appState.currentViewState = viewState;
          }
        });
        // Update currentViewState with the new initialViewState
        appState.currentViewState = initialViewState;
      }

      // Load geo_json_data.json
      appState.zipContent
        .file(`${sweepId}/${runId}/geo_json_data.json`)
        .async('string')
        .then(text => {
          const geoJsonData = JSON.parse(text);
          const features = geoJsonData.features;

          // Categorize features
          const paths = [];
          const totalMeasuredAreas = [];

          features.forEach(feature => {
            const category = feature.properties.category;
            const step = feature.properties.current_step_in_animation;

            // Only add path features for the current run (using runId from loadData parameter)
            if (category === 'path_of_episode' && feature.properties.run_id === runId) {
              paths[step] = feature.geometry.coordinates;
            } else if (category === 'target_area') {
              totalMeasuredAreas[step] = feature.geometry.coordinates;
            }
          });

          appState.paths = paths;
          appState.totalMeasuredAreas = totalMeasuredAreas;

          // Adjust colorsPerTimestep length to be the number of positions
          appState.colorsPerTimestep = new Array(paths.length).fill(null);

          slider.max = appState.colorsPerTimestep.length - 1;

          // Load metric_data.json
          appState.zipContent
            .file(`${sweepId}/${runId}/metric_data.json`)
            .async('string')
            .then(text => {
              appState.metricData = JSON.parse(text);

              // Load radiation_point_cloud.json
              appState.zipContent
                .file(`${sweepId}/${runId}/radiation_point_cloud.json`)
                .async('string')
                .then(text => {
                  appState.pointClouds = JSON.parse(text);

                  // Update layers function assigned to appState
                  appState.updateLayers = function(index) {
                    // Path for the current index
                    const pathPoints = appState.paths[index];
                    
                    // Process target_area depending on whether it is a multipolygon or not
                    let currentTotalMeasuredData = [];
                    const target = appState.totalMeasuredAreas[index];
                    if (target) {
                      let polygons = [];
                      // Check if target is actually a Polygon (an array of rings) or MultiPolygon (an array of polygons)
                      if (target.length > 0 && Array.isArray(target[0]) && target[0].length > 0 && Array.isArray(target[0][0])) {
                        if (typeof target[0][0][0] === 'number') {
                          // It is a Polygon: an array of rings, wrap it into an array
                          polygons = [target];
                        } else {
                          // It is a MultiPolygon: each element is an array of rings
                          polygons = target;
                        }
                      }
                      polygons.forEach(poly => {
                        currentTotalMeasuredData.push({ coordinates: poly });
                      });
                    }
                    
                    // Create layers as before using the new currentTotalMeasuredData
                    const pathLayer = new deck.PathLayer({
                      id: 'path-layer',
                      data: [{ path: pathPoints }],
                      getPath: d => d.path,
                      getColor: hexToRgb(pathColorInput.value),
                      widthMinPixels: parseInt(pathSizeInput.value),
                      capRounded: false
                    });
                    
                    const totalMeasuredAreaGeoJsonLayer = new deck.GeoJsonLayer({
                      id: 'geojson-layer',
                      data: {
                        type: 'FeatureCollection',
                        features: currentTotalMeasuredData.map(polygon => ({
                          type: 'Feature',
                          geometry: {
                            type: 'Polygon',
                            coordinates: polygon.coordinates
                          },
                          properties: {}
                        }))
                      },
                      getFillColor: [20, 225, 230, 100],
                      getLineColor: [255, 255, 255],
                      getLineWidth: 1,
                      stroked: false,
                      filled: true,
                      lineWidthMinPixels: 1,
                      extensions: [
                        new deck._TerrainExtension({
                          terrainDrawMode: 'drape'
                        })
                      ]
                    });
                    
                    const pointCloudData = appState.pointClouds[index];
                    const pointCloudLayer = new deck.PointCloudLayer({
                      id: 'point-cloud-layer',
                      data: pointCloudData.positions.map((pos, i) => ({
                        position: pos,
                        color: pointCloudData.colors[i]
                      })),
                      getPosition: d => d.position,
                      getColor: d => d.color,
                      pointSize: parseInt(bubbleSizeInput.value),
                      extensions: [
                        new deck._TerrainExtension({
                          terrainDrawMode: 'drape'
                        })
                      ]
                    });
                    
                    appState.deckGL.setProps({
                      layers: [terrainLayer, totalMeasuredAreaGeoJsonLayer, pathLayer, pointCloudLayer]
                    });
                    
                    // Update metric info
                    metricContainer.innerHTML = '';
                    for (const [key, values] of Object.entries(appState.metricData)) {
                      const metricElement = document.createElement('div');
                      metricElement.className = 'metric-item';
                      metricElement.innerHTML = `<span>${key}:</span><span>${values[index]}</span>`;
                      metricContainer.appendChild(metricElement);
                    }
                  };
                  // Call updateLayers with initial index 0
                  appState.updateLayers(0);
                })
                .catch(error => {
                  console.error('Error loading radiation_point_cloud:', error);
                });
            })
            .catch(error => {
              console.error('Error loading metric_data:', error);
            });
        })
        .catch(error => {
          console.error('Error loading geo_json_data:', error);
        });
    })
    .catch(error => {
      console.error('Error loading center coordinates:', error);
    });
}

function hexToRgb(hex) {
  const bigint = parseInt(hex.slice(1), 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = (bigint & 255);
  return [r, g, b];
}