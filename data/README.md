# Data

`hour.csv` is the hourly file from the UCI Bike Sharing dataset, committed unchanged.

- Source: [UCI Machine Learning Repository, Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- Citation: Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)

## Columns

| Column | Meaning | Used by the models? |
|---|---|---|
| `instant` | row index | no |
| `dteday` | date | no (used for the time-based split and the dashboard) |
| `season` | 1 winter, 2 spring, 3 summer, 4 fall | yes |
| `yr` | 0 = 2011, 1 = 2012 | yes |
| `mnth` | month (1 to 12) | yes |
| `hr` | hour (0 to 23) | yes |
| `holiday` | 1 if the day is a holiday | yes |
| `weekday` | day of the week | yes |
| `workingday` | 1 if the day is neither a weekend nor a holiday | yes |
| `weathersit` | 1 clear, 2 mist, 3 light rain or snow, 4 heavy rain or snow | yes (4 is merged into 3) |
| `temp` | temperature, normalized: (°C + 8) / 47 | yes |
| `atemp` | "feels like" temperature, normalized | no (almost the same as `temp`) |
| `hum` | humidity / 100 | yes |
| `windspeed` | wind speed / 67 | yes |
| `casual` | rentals by casual users | no (part of the target) |
| `registered` | rentals by registered users | no (part of the target) |
| `cnt` | total rentals in the hour (`casual + registered`) | target |

The cleaning is in `src/data.py`.
