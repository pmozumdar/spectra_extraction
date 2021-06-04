"""

echelle2d.py

A generic set of code to process echelle 2d spectra that are stored in a
multi-extension fits format.  This code is used by some of the ESI and NIRES
code that are in the keckcode repository

"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from astropy.io import fits as pf
from astropy.table import Table
from astropy.modeling import models
from specim_test.specim import imfuncs as imf
from specim_test.specim import specfuncs as ss


# ---------------------------------------------------------------------------

class Ech2d(list):
    """

    A generic class to process echelle 2d spectra that are stored in a
    multi-extension fits format.  This code is used by some of the ESI
    and NIRES code that is in the keckcode repository

    """

    def __init__(self, inspec, varspec=None, hdustart=1, ordinfo=None,
                 logwav=False, fixnans=True, summary=True, verbose=True,
                 debug=False):
        """

        Initialized the class by reading in the input spectra.

        These spectra are designated by the inspec parameter, as well as
         by the optional varspec parameter if there is a separate
         container for the variance spectra
        The inspec paramter can be of two forms:

          1. The name of an input file containing the echelle spectra in
             multiple extensions

          2. A HDUList object in which the echelle spectra are in the
             associated HDUs in the list

        """

        """ Initialize some variables """
        self.infile = None
        self.plotgrid = None
        self.p0hdr = None

        """
        Set the input HDU(s) in a manner reflecting how the input spectra
         are provided
        """
        hdu = self.set_hdulist(inspec, debug)
        if varspec is not None:
            tmpname = self.infile
            varhdu = self.set_hdulist(varspec, debug)
            self.infile = tmpname
        else:
            varhdu = None

        """ Load each order into its own Spec2d container """
        for i in range(hdustart, len(hdu)):
            tmpspec = ss.Spec2d(hdu, hext=i, invar=varhdu, logwav=logwav,
                                fixnans=fixnans, verbose=False)
            tmpspec.spec1d = None
            self.append(tmpspec)

        """
        Set up a default information table if none has been provided
        """
        if ordinfo is not None:
            self.ordinfo = ordinfo
        else:
            order = np.arange(len(self), dtype=int)
            pixmin = np.zeros(len(self), dtype=int)
            pixmax = np.ones(len(self), dtype=int) * -1
            self.ordinfo = Table([order, pixmin, pixmax], 
                                 names = ['order', 'pixmin', 'pixmax'])

        """
        If the first data HDU is not 0, then save the PHDU header, since
        it may have useful information about, e.g., the observations or
        WCS information
        """
        if hdustart > 0:
            self.p0hdr = hdu[0].header.copy()

        """ Summarize the input if requested """
        if summary:
            self.input_summary()

    # -----------------------------------------------------------------------

    def set_hdulist(self, spec, verbose=True):
        """

        Gets the HDUList based on the given spec parameter, which can be
        either a string (the input filename) or an already existing HDUList
      
        """
        if isinstance(spec, str):
            try:
                hdu = pf.open(spec)
            except IOError:
                raise IOError('ERROR: %s not found' % spec)
            self.infile = spec

        elif isinstance(spec, pf.HDUList):
            hdu = spec

        """ Give some information and return the HDUList """
        if verbose:
            hdu.info()
        return hdu

    # -----------------------------------------------------------------------

    def input_summary(self):
        """

        Prints a summary of the data in the input file

        """

        print('Order  Shape    Dispaxis')
        print('----- --------- --------')


        """ Get information on the order labels, if it exists """
        if self.ordinfo is not None:
            orders = self.ordinfo['order']
        else:
            orders = np.arange(len(self))

        """ Print out the information for each order """
        for spec, order in zip(self, orders):
            print(' %2d   %dx%d     %s' % 
                  (order, spec.data.shape[1], spec.data.shape[0],
                   spec.dispaxis))

    # --------------------------------------------------------------------

    def plot_2d(self, fillval=0., gap=50, **kwargs):
        """

        Plots in one figure the 2-d spectra from all of the orders.
        These are stored in separate HDUs in the input file

        """

        """
        Make a temporary array to hold all the spectra
        """
        ycurr = 0
        arrinfo = np.zeros((len(self), 3), dtype=int)
        for i, spec in enumerate(self):
            arrinfo[i, 0] = spec.npix
            if i == 0:
                ycurr = 0
            else:
                ycurr += gap
            arrinfo[i, 1] = ycurr
            ycurr += spec.nspat
            arrinfo[i, 2] = ycurr
        aimax = arrinfo.max(axis=0)
        fulldat = np.zeros((aimax[2], aimax[0])) + fillval

        """
        Loop through the orders and put the 2d spectra into the temporary
        array
        """
        for i, spec in enumerate(self):
            xmax = arrinfo[i, 0]
            ystart = arrinfo[i, 1]
            yend = arrinfo[i, 2]
            fulldat[ystart:yend, 0:xmax] = spec.data.copy()

        """ Display the data """
        tmpim = imf.Image(pf.PrimaryHDU(fulldat))
        tmpim.display(mode='xy', **kwargs)

        """ Mark limits of "good" data if available """
        count = 0
        for spec, info in zip(self, self.ordinfo):
            tmp = np.arange(spec.npix)
            ystart = arrinfo[count, 1]
            yend = arrinfo[count, 2]
            xstart = tmp[info['pixmin']]
            xend = tmp[info['pixmax']]
            tmpx = np.array([xstart, xstart])
            tmpy = np.array([ystart, yend])
            plt.plot(tmpx, tmpy, color='g', lw=3)
            tmpx = np.array([xend, xend])
            plt.plot(tmpx, tmpy, color='g', lw=3)
            count += 1
        #     
        # 
        # # plt.figure(figsize=(10,10))
        # plt.subplots_adjust(hspace=0.001)
        # fig = plt.gcf()
        # for spec, info in zip(self, self.ordinfo):
        #     tmp = np.arange(spec.npix)
        #     B = tmp[info['pixmin']]
        #     R = tmp[info['pixmax']]
        #     axi = fig.add_subplot(10, 1, info['order'])
        #     spec.display(hext=(i+1), mode='xy', axlabel=False, **kwargs)
        #     plt.axvline(B, color='g', lw=3)
        #     plt.axvline(R, color='g', lw=3)
        #     plt.setp(axi.get_xticklabels(), visible=False)
        #     axi.set_xlabel('', visible=False)

    # --------------------------------------------------------------------

    def plot_profiles(self, bgsub=True, showfit=False, fitrange=None,
                      showap=True, xunits='default', maxx=140., fontsize=12,
                      verbose=True, **kwargs):
        """

        Plots, in one figure, the spatial profiles for all the 10 orders

        Optional inputs
          bgsub   - Set to true (the default) if the data have had the sky
                     subtracted already at this point.  If this is the case,
                     then the "sky" level in each profile should be close to
                     zero and, therefore, the subplots can be displayed in a
                     way that does not require y-axis labels for each profile
          showfit - Show the fit to the profiles? Default=False
        """

        if bgsub:
            normspec = True
            plt.subplots_adjust(wspace=0.001)
        else:
            normspec = False
        plt.subplots_adjust(hspace=0.001)

        """ Set up the figure and the full-sized frame for the final labels """
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        """ Set up a grid for plotting """
        if self.plotgrid is not None:
            grid = self.plotgrid
        else:
            grid = (1, len(self))

        """ See if we can convert the x axes to arcsecs """
        if xunits == 'default':
            if 'pixscale' in self.ordinfo.colnames:
                xunits = 'arcsec'
            else:
                xunits = 'pix'
        elif xunits == 'arcsec':
            if 'pixscale' not in self.ordinfo.colnames:
                print('Warning: no pixel scale information. Plotting in pix')
                xunits = 'pix'
                
        """ Loop through and plot the profiles """
        if verbose:
            print('Plotting spatial profiles')
            print('-------------------------')
        count = 1
        for spec, info in zip(self, self.ordinfo):
            if self.ordinfo is not None:
                pixrange = [info['pixmin'], info['pixmax']]
            else:
                pixrange = None
            axi = fig.add_subplot(grid[0], grid[1], count)
            if showfit:
                mod = spec.p0
            else:
                mod = None
            if xunits == 'arcsec':
                pixscale = info['pixscale']
            else:
                pixscale = None
            spec.spatial_profile(normalize=normspec, title=None, model=mod,
                                 pixrange=pixrange, pixscale=pixscale,
                                 fontsize=fontsize, verbose=verbose,
                                 ax=axi, fig=fig, **kwargs)
            if xunits == 'pix':
                plt.xlim(-1, maxx)
            if normspec:
                plt.ylim(-0.1, 1.1)
            if count == 1 or count == (grid[1] + 1):
                pass
            else:
                plt.setp(axi.get_yticklabels(), visible=False)
                axi.set_ylabel('', visible=False)
            if grid[0] > 1 and count < (grid[0]-1)*grid[1]:
                plt.setp(axi.get_xticklabels(), visible=False)
            axi.set_xlabel('', visible=False)
            axi.annotate('%d' % info['order'], (0.05, 0.9),
                         xycoords='axes fraction')
            count += 1

        if self.infile is not None:
            ax.set_title('Spatial Profiles for %s' % self.infile)
        else:
            ax.set_title('Spatial Profiles')
        ax.set_xlabel('Spatial Direction (%s)' % xunits, fontsize=fontsize)
        ax.xaxis.set_label_coords(0.5, -0.05)
        ax.yaxis.set_label_coords(-0.03, 0.5)

    # --------------------------------------------------------------------
    def plot_fitted_model(self, columns=3, title=None, fig=None):
        """
        Plots, in one figure, the fitted models for all the orders provided
        a model for that order is avaialble.
        """
        """First check which orders have a model to fit"""
        order = []
        spec_list = []
        spec_with_mod = 0
        for spec, info in zip(self, self.ordinfo):
            if spec.mod0 is not None:
                spec_list.append(spec)
                order.append(info['order'])
                spec_with_mod +=1

        columns = columns
        rows = (spec_with_mod // columns) + 1
        if (spec_with_mod % columns)==0:
            rows -= 1
            
        """setting up the figure and gridspec """
        if fig is not None:
            fig = fig
        else:
            fig = plt.figure(figsize=(25, 20))
        gs = gridspec.GridSpec(rows, columns, figure=fig)

        for i,spec in enumerate(spec_list):

            gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i],
                                               hspace=0, height_ratios=[3,1,3])
            profile = spec.profile
            mod_y = spec.mod0(profile.x)
            diff = profile.y - mod_y

            ax1 = fig.add_subplot(gs_sub[0, 0])
            ax1.set_xticks([])
            ax1.plot(profile.x, profile.y, color='b', linestyle='solid',
                                   drawstyle='steps', label='Spatial profile')
            ax1.plot(profile.x, mod_y, color='g', drawstyle='steps',
                                                            label='model fit')
            ax1.annotate('order:%d' %order[i], (0.05, 0.9),
                                        xycoords='axes fraction', fontsize=12)
            ax1.legend()
            ax1.set_ylabel('Relative Flux')
            
            ax2 = fig.add_subplot(gs_sub[1, 0], sharex=ax1)
            ax2.plot(profile.x, diff, 'r', drawstyle='steps')
            ax2.hlines(y=0, xmin=min(profile.x), xmax=max(profile.x))
            ax2.set_ylabel('Difference')
            
            ax3 = fig.add_subplot(gs_sub[2, 0])
            ax3.plot(profile.x, profile.y, color='b', linestyle='solid',
                                    drawstyle='steps', label='Spatial profile')
            label_g = True
            label_m = True
            for i, md in enumerate(spec.mod0):
                mod_y = md(profile.x)
                if isinstance(md, models.Gaussian1D):
                    if label_g:
                        plt.plot(profile.x, mod_y, color='k',
                                     drawstyle='steps', label='Gaussian prof.')
                        label_g = False
                    else:
                        plt.plot(profile.x, mod_y, color='k', drawstyle='steps')
                elif isinstance(md, models.Moffat1D):
                    if label_m:
                        plt.plot(profile.x, mod_y, color='r', drawstyle='steps',
                                                           label='Moffat prof.')
                        label_m = False
                    else:
                        plt.plot(profile.x, mod_y, color='r', drawstyle='steps')
            ax3.legend()
            ax3.set_xlabel('Spatial direction (pix)')
            ax3.set_ylabel('Relative Flux')
        plt.suptitle('Model fit to Spatial Profile', fontsize=16)    
        plt.show()
    # --------------------------------------------------------------------

    def save(self, outfile, mode='input'):
        """

        Saves the spectra in a multi-extension fits format

        """

        """ Start the HDUList that will be saved """
        hdu = pf.HDUList(pf.PrimaryHDU())

        """ Loop through the orders, saving each one to a separate HDU """
        for spec, info in zip(self, self.ordinfo):
            extname = 'Order%02d' % info['order']
            if mode == 'skysub':
                ihdu = pf.ImageHDU(spec.skysub, name=extname)
            else:
                ihdu = pf.ImageHDU(spec, name=extname)
            hdu.append(ihdu)

        """ Save the HDUList to the requested output file """
        print('Saving echelle spectrum to %s' % outfile)
        hdu.writeto(outfile, overwrite=True)
