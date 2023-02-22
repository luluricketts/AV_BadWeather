import requests
import datetime
import numpy as np



if __name__ == "__main__":

    token = 'cWioBKuaYnjAPNTucOrZbYFDOIUvTqtX'
    base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
    endpoint = 'stations'
    latlong = np.array([40.246568, -80.211979])
    bound = 0.1
    header = {'token': token}
    startdate = '2022-12-21'
    enddate = '2022-12-23'
    payload = {
        'startdate': startdate,
        'enddate': enddate,
        'extent': f'{latlong[0] - bound},{latlong[1] - bound},{latlong[0] + bound},{latlong[1] + bound}'
    }

    res = requests.get(
        base_url + 'stations', 
        headers=header,
        params=payload
    ).json()
    print(res)


    dists = np.empty(len(res['results']))
    for i,st in enumerate(res['results']):
        station_latlong = np.array([st['latitude'], st['longitude']])
        dists[i] = np.sqrt(np.sum((latlong - station_latlong) ** 2))

    stationId = res['results'][np.argmin(dists)]['id']
    print(stationId)
    
    base_url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/'
    payload = {
        'datasetid': 'GHCND',
        'stationid': stationId,
        'startdate': startdate,
        'enddate': enddate,
        'units': 'standard'
    }
    res_station = requests.get(
       base_url + 'data',
        headers=header,
        params=payload
    ).json()
    print(res_station)

    # print(res_station)
    # r = requests.get(base_url + 'data?datasetid=GHCND', 
    #     headers=header,
    #     params = {'datasetid': 'GHCND'})
    # print(r)

    # r = requests.get('https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND',
    #                    headers={'token':token})
    # print(r.json())



    # wfo = res['properties']['gridId']
    # x = res['properties']['gridX']
    # y = res['properties']['gridY']
    # print(wfo, x, y)
    # res = requests.get(f'https://api.weather.gov/gridpoints/{wfo}/{x},{y}/stations').json()
    # dists = np.empty(len(res['features']))
    # for i,st in enumerate(res['features']):
       
    #     station_latlong = np.flip(np.array(st['geometry']['coordinates']))
    #     dists[i] = np.sqrt(np.sum((latlong - station_latlong)**2))
    
    # # print(res['features'][np.argmin(dists)])
    # # print(res['features'][np.argmin(dists)])
    # stationId = res['features'][np.argmin(dists)]['properties']['stationIdentifier']
    # # time = str(datetime.datetime.now()).replace(' ', 'T') + 'Z'
    # # time = str(datetime.datetime(2020, 12, 12, 12, 0, 0, 0)).replace(' ', 'T') + 'Z'
    # time = '2023-01-30T16:56:00+00:00'
    # date = time.split('T')[0]
    # print(time)
    # # res_station = requests.get(f'https://api.weather.gov/stations/{stationId}/observations/{time}').json()

    # res_station = requests.get(f'https://api.weather.gov/stations/{stationId}/tafs/{date}/{time}').json()

    # print(res_station)

