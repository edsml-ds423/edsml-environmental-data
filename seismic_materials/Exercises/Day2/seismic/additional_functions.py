import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import urllib
import cv2
# from win32api import GetSystemMetrics

__all__ = ['read_img', 'remove_bounds', 'plot', 'reflection_time', 'sample_trace', 'nmo_correction', 'click', 'vel_curve', 'semblance']

def read_img (location, visualise=True):
    """
    reads in image and transforms it into a 2-bit array
    
    parameters
    -------------
    location: str
        string containing the file location
    visualise: boolean
        decide whether to visualise the resulting array
    
    returns
    --------------
    img: numpy.ndarray
        2-bit 2D numpy array containing geological model

    """
    # read in image
    img=mpimg.imread(location)
    assert type(img) == np.ndarray, 'data not in numpy array format'
    
    if np.shape(img)[-1] == 3: # if image 3 bands (RGB)
        # transform to 1 band
        img = (img[:,:,0] + img[:,:,1] + img[:,:,2]) /3
    
    if visualise == True:
        # plot geolgical model
        plt.title ('Thin wedge model')
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        plt.imshow(img, cmap = 'binary')
        plt.show()
    
    #check if binary
    assert len(set(img.flatten())) == 2, 'image is not binary'
    
    return img

def remove_bounds(model, visualise = True):
    """
    removes 2-bit boundary between geological layers from the original image
    
    parameters
    -------------
    model: numpy.ndarray
        numpy array containing the geological model with boundaries between layers
    visualise: boolean
        decide whether to visualise the resulting array
    
    returns
    --------------
    model: numpy.ndarray
        2D numpy array containing geological model with each geological section labelled

    """
    # sub out all boundaries
    for j in range(np.shape(model)[1]):
        for i in range(np.shape(model)[0]):
            if model[i,j] == 0:
                model[i][j] = model [i-1,j]
    if visualise == True:
        plt.title ('sectioned geological cross section')
        plt.imshow(model)
        plt.colorbar()
        plt.show()
        
    return model-1 #scale back all values now that bounds are removed

def plot(cmp, noffsets, nsamples, dt):
    """
    Plots synthetic cmp gathers
    """
    cutoff = 0.1
    plt.imshow(cmp, extent=[0.5, noffsets + 0.5, dt*nsamples, 0], 
              aspect='auto', cmap='Greys', vmin=-cutoff, vmax=cutoff, 
              interpolation='none')
    
    # following is purely for visual purposes
    trace_numbers = list(range(1, noffsets + 1))
    plt.xticks(trace_numbers)
    plt.title('CMP')
    plt.xlabel('Trace number')
    plt.ylabel('Time (s)')
    plt.show()

def reflection_time(t0, x, vnmo):
    """
    Calculate the travel-time of a reflected wave.
    
    Doesn't consider refractions or changes in velocity.
        
    The units must be consistent. E.g., if t0 is seconds and
    x is meters, vnmo must be m/s.
    
    Parameters
    ----------
    
    t0 : float
        The 0-offset (normal incidence) travel-time.
    x : float
        The offset of the receiver.
    vnmo : float
        The NMO velocity.
        
    Returns
    -------
    
    t : float
        The reflection travel-time.
        
    """
    t = np.sqrt(t0**2 + x**2/vnmo**2)
    return t

def sample_trace(trace, time, dt):
    """
    Sample an amplitude at a given time using interpolation.
    
    Parameters
    ----------
    
    trace : 1D array
        Array containing the amplitudes of a single trace.
    time : float
        The time at which I want to sample the amplitude.
    dt : float
        The sampling interval
        
    Returns
    -------
    
    amplitude : float or None
        The interpolated amplitude. Will be None if *time*
        is beyond the end of the trace or if there are less
        than 2 points between *time* and the end.
        
    """
    # The floor function will give us the integer
    # right behind a given float.
    # Use it to get the sample number that is right
    # before our desired time.
    before = int(np.floor(time/dt))
    N = trace.size # 1D array
    # Use the 4 samples around time to interpolate
    samples = np.arange(before - 1, before + 3)
    if any(samples < 0) or any(samples >= N):
        amplitude = None
    else:
        times = dt*samples # list of times
        amps = trace[samples] #list of amplitudes
        interpolator = CubicSpline(times, amps)
        amplitude = interpolator(time)
    return amplitude

def nmo_correction(cmp, dt, offsets, velocities):
    """
    Performs NMO correction on the given CMP.
    
    The units must be consistent. E.g., if dt is seconds and
    offsets is meters, velocities must be m/s.
    
    Parameters
    ----------
    
    cmp : 2D array
        The CMP gather that we want to correct.
    dt : float
        The sampling interval.
    offsets : 1D array
        An array with the offset of each trace in the CMP.
    velocities : 1D array
        An array with the NMO velocity for each time. Should
        have the same number of elements as the CMP has samples.
        
    Returns
    -------
    
    nmo : 2D array
        The NMO corrected gather.
        
    """
    nmo = np.zeros_like(cmp) #set array for corrected data
    nsamples = cmp.shape[0] # samples along trace
    times = np.arange(0, nsamples*dt, dt) # list from 0 to total time, spaced by each timestep
    for i, t0 in enumerate(times): # index, timestep
        for j, x in enumerate(offsets): # index, offset
                             # (travel-time, receiver offset, correction velocity)
            t = reflection_time(t0, x, velocities[i]) # find the travel-time for a specific reflection on a trace
                                    # (trace j, reflection time, timestep)
            # amplitude = sample_trace(cmp[:, j], t, dt) #get amplitude from across a few samples
            
            # works too but not as tidy
            try: amplitude = cmp[:, j][int(t/dt)] #sample_trace(cmp[:, j], t, dt) #get amplitude from across a few samples
            except: amplitude = None
            
            # If the time t is outside of the CMP time range,
            # amplitude will be None.
            if amplitude is not None:
                nmo[i, j] = amplitude
    return nmo


def click(amp_store, v_test, times, resize = [540,960]):
    """
    allows to use a mouse to recover values from the semblance plot
    """
    #the [x, y] for each right-click event will be stored here
    right_clicks = []
    #this function will be called whenever the mouse is right-clicked
    def mouse_callback(event, x, y, flags, params):
        right_clicks
        #right-click event value is 2
        if event == 2:
            #global right_clicks

            #store the coordinates of the right-click event
            right_clicks.append([x, y])

            #this just verifies that the mouse data is being collected
            #you probably want to remove this later
            print (right_clicks)

    img = np.swapaxes(amp_store,0,1 )   
    imS = cv2.resize(img, (resize[0], resize[1])) 

    scale_width = 1000 / imS.shape[1]
    scale_height = 600 /imS.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(imS.shape[1] * scale)
    window_height = int(imS.shape[0] * scale)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    #set mouse callback function for window
    cv2.setMouseCallback('image', mouse_callback)
    
    img = img.clip(0, np.max(img))
    
    img_n = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imC = cv2.applyColorMap(img_n, cv2.COLORMAP_JET)
    imS = cv2.resize(imC, (resize[0], resize[1])) 
    
    cv2.imshow('image', imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
     # convert obtained point indices of resized image to original indices and recover time/velocity values
    NMO_val = []
    for i in right_clicks:
        vel_index = int(i[0]/resize[0] * np.shape(amp_store)[0])
        time_index = int(i[1]/resize[1] * np.shape(amp_store)[1])

        NMO_val.append([int (v_test[vel_index]), time_index])
        print('NMO_vel= ', int (v_test[vel_index]), '| time = ', times[time_index])
    
    return NMO_val

def vel_curve(amp_store, NMO_val, times):
    """
    Create piecewise linear curve using obtained velocity data points
    """
    
    # first set of values is assumed to be a constant velocity up until our first pick
    v_nmo = np.ones([NMO_val[0][1]]) * NMO_val[0][0]
    
    #rotating data for simpler intuition
    img = np.swapaxes(amp_store,0,1 )

    # create linear function between each velocity point pair
    for i in range(len(NMO_val)-1):
        times_sub = times[NMO_val[i][1]:NMO_val[i+1][1]]

        v1 = NMO_val[i][0]
        v2 = NMO_val[i+1][0]
        t1 = times[NMO_val[i][1]]
        t2 = times[NMO_val[i+1][1]]

        v_nmo_sub = v1 + ((v2 - v1)/(t2 - t1))*(times_sub-t1)
        v_nmo = np.append(v_nmo, v_nmo_sub)

    # add constant velocity for remainder data at depth
    v_nmo = np.append(v_nmo, np.ones(np.shape(img)[0] - NMO_val[-1][1]) * NMO_val[-1][0])
    
    # visualise velocity profile
    plt.plot(v_nmo, times)
    plt.xlabel("velocity (m/s)")
    plt.ylabel("two way time (s)")
    plt.gca().invert_yaxis()
    plt.title('velocity profile')
    plt.show()

    return v_nmo

def semblance(v_test, cmp, dt, offsets, times, verbose = True): 
    """create semblance plot"""
    
    # store outputs
    amp_store = [] # ideally we would use a numpy array here!

    for count, v in enumerate(v_test): #iteratively try different velocities
        v_nmo = np.ones(len(times))*v # constant velocity array for correction

        nmo = nmo_correction(cmp, dt, offsets, v_nmo) # NMO correction

        sum_row = np.sum(nmo, axis = 1) # row summation across traces
        amp_store.append(sum_row)
        if verbose and count%1 ==0: print ('progress = ', ((count+1)/len(v_test))*100, '%')
    return amp_store 