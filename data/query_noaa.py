import requests
import numpy as np
from datetime import datetime
import time


class NOAA_API:

    def __init__(self, token='cWioBKuaYnjAPNTucOrZbYFDOIUvTqtX'):
        self.base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
        self.header = {'token': token}


    def try_request(self, endpoint, payload):
        return requests.get(
            self.base_url + endpoint, 
            headers = self.header,
            params = payload
        )

    def make_query(self, endpoint, payload):

        res = self.try_request(endpoint, payload)
        if res.status_code != 200:
            if res.status_code == 429: # limit reached
                now = datetime.now().strftime("%H:%M:%S")
                print(f'NOAA API Limit Reached at {now}, trying again in 1 hour')
                time.sleep(60 * 60)
                return self.make_query(endpoint, payload)
            else:
                return False

        res = res.json()
        if res == {}:
            return False
        return res



def query_noaa_bad_weather(date, latlong):
    """
    Queries national weather service for a date + location
    returns True if there is ANY bad weather present for the config, else False
    note: api limit 5 queries/sec, 10,000 per day

    date: str (YYYY-MM-DD)
    latlong: np.array [lat, long]
    returns bool success and label
    """
    
    def get_obs_values():
        # https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/doc/GHCND_documentation.pdf
        keys = ['PRCP', 'SNOW', 'SNWD'] + [f'WT0{i}' for i in range(1, 10)] + [f'WT{i}' for i in range(10, 23)]
        vals = ['rain', 'snow', 'snow', 'fog', 'fog', 'rain', 'snow', 'snow', 'snow', 'fog',
                'fog', 'snow', 'fog', 'fog', 'rain', 'rain', 'rain', 'rain', 'rain', 'rain', 'snow',
                'rain', 'rain', 'fog', 'fog']
        return dict(zip(keys, vals))

    noaa = NOAA_API()

    bound = 0.1
    payload = {
        'startdate': date,
        'enddate': date,
        'extent': f'{latlong[0] - bound},{latlong[1] - bound},{latlong[0] + bound},{latlong[1] + bound}'
    }
    res = noaa.make_query('stations', payload)
    if not res: return False, None, None

    dists = np.empty(len(res['results']))
    for i,st in enumerate(res['results']):
        station_latlong = np.array([st['latitude'], st['longitude']])
        dists[i] = np.sqrt(np.sum((latlong - station_latlong) ** 2))
    stationId = res['results'][np.argmin(dists)]['id']

    payload = {
        'datasetid': 'GHCND',
        'stationid': stationId,
        'startdate': date,
        'enddate': date,
        'units': 'metric'
    }
    res_station = noaa.make_query('data', payload)
    if not res_station: return False, None, None
    
    obs_values = get_obs_values()
    thres = 2 # mm
    for r in res_station['results']:
        if r['datatype'] in obs_values:
            if r['value'] >= thres:
                return True, obs_values[r['datatype']], r['value']
    
    return False, None, None





if __name__ == "__main__":

    startdate = '2023-02-11'
    latlong = np.array([40.246568, -80.211979])

    res = query_noaa_bad_weather(startdate, latlong)
    print(res)