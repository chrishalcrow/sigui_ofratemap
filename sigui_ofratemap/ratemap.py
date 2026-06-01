from spikeinterface_gui.view_base import ViewBase
import pynapple as nap
import numpy as np
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg

class RateMapView(ViewBase):
    # Unique identifier for this view (used in layouts)
    id = "ratemap"
    
    _supported_backend = ['qt']
    
    # Help text displayed to users
    _gui_help_txt = "My custom view that displays unit information"

    def __init__(self, controller=None, parent=None, backend="qt"):
        ViewBase.__init__(self, controller=controller, parent=parent,  backend=backend)

    _settings = [
            {'name': 'smooth_sigma', 'type': 'float', 'value': 0},
            {'name': 'bins', 'type': 'int', 'value': 40},
            {'name': 'moving', 'type': 'float', 'value': 0}
        ]
    
    def _qt_make_layout(self):

        self.estimated_position = [np.nan, np.nan]
        self.current_cross = None

        external_data = self.controller.external_data
        position = external_data['position']
        self.position = position

        from spikeinterface_gui.myqt import QT
        import pyqtgraph as pg

        self.layout = QT.QVBoxLayout()

        self.graph_canvas = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_canvas)

        self.on_settings_changed()#this do refresh
        self._qt_refresh()

    def on_settings_changed(self):
     
        self.colormap = pg.colormap.get('CET-L9')
        self.lut = self.colormap.getLookupTable(0.0, 1.0)
        

    def _qt_refresh(self):
        """Update the view with current data (called when data changes)"""
        
        self.graph_canvas.clear()
        
        visible_unit_ids = self.controller.get_visible_unit_ids()
        n = len(visible_unit_ids)

        bins = self.settings['bins']
        smooth_sigma = self.settings['smooth_sigma']
        moving = self.settings['moving']

        for i in range(n):

            unit_id = visible_unit_ids[i]

            spike_inds = self.controller.get_spike_indices(unit_id, segment_index=0)
            spikes = self.controller.spikes[spike_inds]['sample_index']/30_000

            position_x = self.position['x']
            position_y = self.position['y']

            dx = np.diff(position_x)
            dy = np.diff(position_y)
            dt = np.diff(position_x.times())

            speed = nap.Tsd(
                        t=position_x.times()[1:],
                        d=np.sqrt( (dx/dt)**2 + (dy/dt)**2),
                    )
            moving = speed.threshold(moving, method='above').time_support

            tuning_curve = nap.compute_tuning_curves(
                nap.TsGroup([spikes]),
                np.stack([position_x, position_y], axis=1),
                range=[(0, 100), (0, 100)],
                bins=bins,
                epochs=moving,
                feature_names=["x", "y"],
            )

            tuning_curve = gaussian_filter_nan(
                tuning_curve,
                sigma=(0, smooth_sigma, smooth_sigma),
                keep=False,
            )

            image = pg.ImageItem(tuning_curve[0].to_numpy(), lut=self.lut)
            self.view = self.graph_canvas.addPlot(title=f"Unit {unit_id}")
            bar = pg.ColorBarItem(colorMap=self.colormap) #default is 25
            bar.setImageItem(image)
            self.view.setDefaultPadding(0)
            self.view.addItem(image)
            self.view.setRange(rect=image.boundingRect(), padding=0)
            self.view.getViewBox().setContentsMargins(0, 0, 0, 0)
            self.graph_canvas.addItem(bar)

    def _qt_on_spike_selection_changed(self):

        if self.current_cross is not None:
            self.view.removeItem(self.current_cross)

        bins = self.settings['bins']

        selected_inds  = self.controller.get_indices_spike_selected()
        spike_sample_index = self.controller.spikes['sample_index'][selected_inds[0]]

        from pynapple import Ts
        
        if len(selected_inds) > 0:
            self.estimated_position = self.position.interpolate(Ts([spike_sample_index/self.controller.sampling_frequency])).values[0]

            self.current_cross = pg.ScatterPlotItem(
                pos=[(self.estimated_position[0]/100*bins, self.estimated_position[1]/100*bins)],
                symbol='x',           # Use '+' for a cross or 'x' for a diagonal cross
                size=20,              # Size in pixels
                pen=pg.mkPen('w', width=2), # 'w' for white, or 'k' for black to contrast your heatmap
                brush=None            # No fill, just the lines
            )
            self.current_cross.setZValue(10)
            self.view.addItem(self.current_cross)
            

        

from scipy.ndimage import gaussian_filter


def gaussian_filter_nan(X, sigma, mode="reflect", keep=True):
    # Check if input is xarray DataArray or Dataset (duck typing)
    is_xarray = hasattr(X, "values") and hasattr(X, "dims") and hasattr(X, "coords")

    # Extract raw numpy array
    data = X.values if is_xarray else X

    V = data.copy()
    V[np.isnan(data)] = 0
    VV = gaussian_filter(V, sigma=sigma, mode=mode, truncate=6)

    W = np.ones_like(data)
    W[np.isnan(data)] = 0
    WW = gaussian_filter(W, sigma=sigma, mode=mode, truncate=6)

    Y = VV / WW
    if keep:
        Y[np.isnan(data)] = np.nan

    if is_xarray:
        # Rebuild xarray with same dims and coords
        import xarray as xr

        return xr.DataArray(Y, dims=X.dims, coords=X.coords, attrs=X.attrs)
    else:
        return Y
