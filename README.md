# Coastal-CFHW-CMSD-Risk-SSP-RCP
Compound Flood-Heatwave Events Elevate Risks to Sustainable Development in China’s Coastal Zones Under Future Scenarios

##Data

  --Coastal provinces' CMSD data for the historical period 2012–2022.
  --Coastal provinces' POP, GDP, and ISA data for the historical period 2012–2022. Among these, POP and GDP data were obtained from the National Bureau of Statistics of China official website. ISA data were derived by calculating the provincial average values of historical ISA raster data based on the vector file of coastal areas.
  --CMSD data under future scenarios were predicted using the Bayesian Structural Time Series (BSTS) model with POP, GDP, and ISA as covariates.
  --POP, GDP, and ISA data under future scenarios were obtained by conducting regional statistical tabulation on existing simulated raster data with the vector file of coastal areas.

## Code Description

  --The CFHW_calculation.py file is used to calculate compound flood-heatwave (CFHW) events (only the code for the historical period is included).
  --The MATLAB script in ERA5 data processing.txt preprocesses temperature and precipitation variables from ERA5 hourly meteorological NetCDF data into a daily dataset.
  --Sen-slope and Mann-kendall test.py estimates the trends of temperature and precipitation variables in coastal areas.
  --BSTS Forecast.py simulates CMSD under future scenarios with POP, GDP, and ISA as covariates.
