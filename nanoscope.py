from __future__ import division
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
from PIL import Image
from matplotlib import cm
import re, os

def split_header_lines(line):
    """ splits header lines into names and values """
    split_line = ['', '']
    if line[:2] == '\*':
        if line == '\*File list':
            split_line = ['Name', 'Value']
        elif line == '\*Ciao image list':
            split_line = ['Channel', '']
        else:
            split_line = [line[2:], '']
    else:
        split_line = line[1:].split(': ')
    return split_line

def separate_header(file):
    """ creates a string array out of the header data """
    file.seek(0)
    end_line = '\*File list end'
    header = []
    channel = 0
    while True:
        line = file.readline().rstrip('\r\n')
        if line == end_line:
            break
        line = split_header_lines(line)
        if line[0] == 'Channel':
            channel += 1
            line[1] = str(channel)
        header.append(line)
    return header

def arrange_data(data, data_shape, sweep):
    """ arranges image data according to size and sweep direction
        assumes channel data is the same shape """
    chan = len(data_shape) #num of chan
    row, col = data_shape[0]
    data = data.reshape((chan*data_shape[0][0], data_shape[0][1]))
    #for i, swp in enumerate(sweep): #might not be needed
    #    if 'Down' in swp:             #all images may need to be flipped
    #        data[i*row:(i+1)*row,:] = data[i*row:(i+1)*row,:][::-1,...]
    return data

def scale_data(data, header, channel_names, data_shape):
    """ use header data to scale data to proper units
        all lengths are in um, heights in nm """
    MAX = 65536.0
    data = data/MAX
    row, col = data_shape[0]
    chan_scale = []
    for i, head in enumerate(header): #search header for relevant info
        if 'Scan size' in head[0]:
            m = re.search('(\d+) nm', head[1])
            if (m != None):
                scan_size = float(m.group(1))/1000.0 #convert to um
        if '@Sens. Zscan' in head[0]:
            height_scale = float(re.search(' (\d+.\d+) ',head[1]).group(1))
            #grab string of numbers not in parentheses
            #not great but it works
        if '@2:Z scale' in head[0]:
            m = re.search(' (\d+.\d+) ', head[1])
            chan_scale.append(float(m.group(1))),

    units = ['None']*len(channel_names)
    for i, name in enumerate(channel_names):
        data[i*row:(i+1)*row,:] *= chan_scale[i]
        if name == 'Height':
            data[i*row:(i+1)*row,:] *= height_scale
            units[i] = 'nm'
        elif name == 'Frequency':
            units[i] = 'Hz'
        elif name == 'Phase':
            units[i] = 'deg'
        elif name == 'Amplitude':
            units[i] = 'V'
    x = np.linspace(0,1,col)*scan_size
    y = np.linspace(0,1,row)*scan_size

    return data, \
           (0, col*scan_size/max(data_shape[0]), 0,row*scan_size/max(data_shape[0])), \
           units #z, extent, units

def parse_channel_data(file, header):
    """ uses information from header to find and parse channel data
        returns some sort of array or dict full of channel data """

    data_shape = []
    data_loc = []
    sweep = []
    channel_names = []
    for i, head in enumerate(header): #search header for relevant info
        if 'Samps/line' in head[0]:
            data_shape.append((int(header[i][1]), int(header[i+1][1])))
        elif 'Data offset' in head[0]:
            #save offset and size in bytes for each chan
            data_loc.append((int(header[i][1]), int(header[i+1][1])))
        elif 'Frame direction' in head[0]:
            sweep.append(head[1])
        elif 'Line direction' in head[0]:
            sweep.append(head[1])
        elif ':Image Data' in head[0]:
            channel_names.append(re.search('"(\w+)"', head[1]).group(1))
    del data_shape[0] #drop first entry, keep only channel specific data
    sweep = zip(*[iter(sweep)]*2)
    file.seek(data_loc[0][0])
    data = np.fromfile(file, dtype=np.int16)
    data = arrange_data(data, data_shape, sweep)
    return data, channel_names, data_shape

    
class AfmData:
    """ This class handles all of the data imports and defines attributes
        like AfmData.data, AfmData.channel_names, this way the object contains
        all of the information needed later for analysis and plotting """
        
    def __init__(self, filename):
        file = open(filename,'rb')
        self.header = separate_header(file)
        self.data, self.channel_names, self.shape = \
            parse_channel_data(file, self.header)
        self.data, self.extent, self.units = \
            scale_data(self.data, self.header, self.channel_names, self.shape)
        file.close()
        self.x = np.linspace(self.extent[0], self.extent[1], self.shape[0][0])
        self.y = np.linspace(self.extent[2], self.extent[3], self.shape[0][1])
    def get_channel_data(self, chan_num):
        """ returns data for only the given channel number, 
            which should be an integer. These numbers are properly scaled. """
        row, col = self.shape[chan_num]
        return self.data[chan_num*row:(chan_num+1)*row,:]
        
#Everything above here is pretty safe.
    
    
#these four functions below need to have more sensible inputs
#they should really just take one dataset as an input
def set_colorscale(data):
    """ sets the colorscale for one data set.
        this could really be much better. """
    return([data.mean()-2.5*data.std(),data.mean()+4.0*data.std()])
    #return np.percentile(data, [0.02, 98.6])

def marker_locate(data, cuts):
    """ this does a pretty good job at tagging markers in 
        height images. """
    dataMax = data.max()
    cuts = [c/100.0 for c in cuts]
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i,j] < cuts[0]*dataMax or data[i,j] > cuts[1]*dataMax: #remember this eliminates anything less than zero as well
                data[i,j] = 0.0
            else:
                data[i,j]=1.0
    return data
    
def threshold_mask(data, limits):
    """ creates a mask for use in flattening images based on some
        threshold values. """
    mask = np.ones(data.shape, dtype=bool)
    for i, row in enumerate(data): #create mask line by line
        cut = np.percentile(row, limits) #brackets define low and high cuts
        ind = np.logical_or(row < cut[0], row > cut[1])
        mask[i, :] = ind
    return mask
    
def create_height_mask(a, limits):
    """ create a threshold mask from the height data. 
        using masks created from other inputs is less reliable. a is an AfmData object"""
    for i, name in enumerate(a.channel_names):
        if name == 'Height':
            row, col = a.shape[i]
            return threshold_mask(a.get_channel_data(i), limits) #could replace with other mask types
    return np.zeros(a.shape[0], dtype = np.bool) #backup in case there is no height data
         
    
def flatten(data, data_shape, order = 1, data_mask = False):
    """ flatten the data Nanoscope-style.
        can be used with any data mask to ignore extreme values."""
    mdata = ma.masked_array(data, mask = data_mask)
    fit = np.zeros(data_shape)
    x = np.arange(0,data_shape[1])
    p = ma.polyfit(x, mdata.transpose(), order).transpose()
    for i, coef in enumerate(p):
        for n, c in enumerate(coef[::-1]):
            fit[i,:] += c*pow(x,n)
    return mdata.data-fit
    
#some easy to use plotting functions
    
def plot_filename(filename):
    """ takes a filename and returns a simple plot of the data channels
        similar to what you would see on the nanoscope software """

    a = AfmData(filename)
        
    plt.rc('text', color = 'white')
    plt.rc('axes', edgecolor = 'white')
    fig, axarr = plt.subplots(1, len(a.channel_names))
    fig.set_size_inches(8.5*len(a.channel_names),8.5)
    fig.patch.set_facecolor('black')
    fig.suptitle(filename, y=0.95, fontsize = 16)
    
    height_mask = create_height_mask(a, [0,95])   #height mask
                                                  #is used to ignore 
                                                  #very tall features
                                                  #that may skew the flatten
                                                  #results
    for i, name in enumerate(a.channel_names):
        data = a.get_channel_data(i)
        row, col = a.shape[i]
        data = flatten(data, a.shape[i], order = 0, data_mask = height_mask)     
        cmin, cmax = set_colorscale(data)
        img = axarr[i].imshow(data, extent=a.extent,
                              origin = 'lower', aspect = row/col, cmap = cm.afmhot,
                              vmin = cmin, vmax = cmax, interpolation = 'gaussian')
                              
        axarr[i].set_title(name)
        axarr[i].grid('on')      
        axarr[i].xaxis.set_ticks_position('bottom')
        axarr[i].yaxis.set_ticks_position('left')
        axarr[i].tick_params(colors = 'white', direction = 'inout')
        cb = plt.colorbar(img, ax = axarr[i], use_gridspec = True,
                          orientation = 'horizontal', fraction = 0.12, 
                          pad = 0.08)
        cb.set_label(a.units[i], color = 'white')
        cb.ax.tick_params(colors = 'white', direction = 'in')
    plt.show()

def save_separate(filename, annotate = False):
    """ Saves a separate file for each channel. """

    dir_name = 'afm_annotated/'
    if os.path.isdir(dir_name)==False: 
        os.mkdir(dir_name[:-1])

    a = AfmData(filename)

    #height_mask = create_height_mask(a, [0,95])   #height mask
                                                  #is used to ignore 
                                                  #very tall features
                                                  #that may skew the flatten
                                                  #results
    for i, name in enumerate(a.channel_names):

        data = a.get_channel_data(i)
        row, col = a.shape[i]
        data = flatten(data, a.shape[i], order = 1)
        cmin, cmax = set_colorscale(data)

        if annotate:
            fdpi = 200
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title(name)
            img = ax.imshow(data, extent=a.extent,
                            origin = 'lower', aspect = row/col, cmap = cm.afmhot,
                            vmin = cmin, vmax = cmax, interpolation = 'gaussian')
            cb = plt.colorbar(img, ax=ax, pad=0.02, shrink=0.8, aspect=20)
            cb.set_label(a.units[i])
        else:
            fdpi = 80
            fsize = tuple([item/fdpi for item in a.shape[i]])
            fig = plt.figure(figsize=fsize, dpi = fdpi)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax = fig.add_subplot(1,1,1)
            img = ax.imshow(data,
                            origin = 'lower', aspect = row/col, cmap = cm.afmhot,
                            vmin = cmin, vmax = cmax, interpolation = 'gaussian')
            ax.axis('off')

        new_filename = os.path.join(dir_name, filename[:-4]+'_'+filename[-3:]+'_'+name+'.png')
        fig.savefig(new_filename, dpi = fdpi)
        plt.close(fig)
        
def height_overlay(filename):
    """ creates an image from an EFM scan with markers from the height scan
        highlighted on top. this assumes a lot of stuff. the data should be
        2 channels, height + Phase/Frequency """
    
    dir_name = 'afm_overlay/'
    if os.path.isdir(dir_name)==False: 
        os.mkdir(dir_name[:-1])

    a = AfmData(filename)
    
    for i, name in enumerate(a.channel_names):
        if name == 'Height':
            height = a.get_channel_data(i)[::-1] #flip image
            row, col = a.shape[i]
            height = flatten(height, a.shape[i], order = 0)
        else:
            efm = a.get_channel_data(i)[::-1] #flip image
            efm = flatten(efm, a.shape[i], order = 0)
            emin, emax = set_colorscale(efm)
            
    #create height alpha data
    heightRGBA = np.zeros((row,col,4)) #nothing here
    heightRGBA[:,:,2] = np.ones((row,col)) #only blue
    heightRGBA[:,:,3] = marker_locate(height, [10, 85]) #alpha data
    im_height = Image.fromarray((heightRGBA*255).astype('uint8'), mode='RGBA')

    #create RGBA data for EFM image
    norm = colors.Normalize(emin, emax) 
    efmRGBA = plt.cm.afmhot(norm(efm)) #create colormap data
    efmRGBA = (efmRGBA*255).astype('uint8') #convert to correct type for Image
    im_efm = Image.fromarray(efmRGBA, mode='RGBA') #create image
    im_efm.paste(im_height, mask = im_height) #combine images
    
    new_filename = os.path.join(dir_name, filename[:-4]+'_'+filename[-3:]+'_combined.png')
    im_efm.save(new_filename, "PNG")