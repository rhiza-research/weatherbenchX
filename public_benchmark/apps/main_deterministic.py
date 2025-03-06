from dash_app import make_app
import xarray as xr
import sys

ds = xr.open_zarr('gs://wb2-app-data/v4/deterministic.zarr')
app = make_app(ds, 'deterministic')
server = app.server

if __name__ == '__main__':
    if sys.argv[1] == 'local':
        app.run_server(debug=True, use_reloader=True)
    else:
        app.run_server(host='0.0.0.0', port=8080, debug=False, use_reloader=False)