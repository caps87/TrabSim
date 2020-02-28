import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import tifffile as tf
from time import strftime, localtime
import csv
import sys
if sys.version_info[0] >= 3:
    from tkinter import *
    from tkinter import ttk
else:
    from Tkinter import *
    import ttk
from PIL import Image, ImageTk
import numpy.random as rnd
import os
import cv2
import pickle

base_path = os.path.join(sys.path[0], 'output/')
  
modelFile = ['model_40_100_80_0.95.sav', 'model_60_40_100_0.94.sav', 'model_60_80_100_0.95.sav',
             'model_20_100_100_0.94.sav', 'model_80_20_80_0.94.sav', 'model_20_60_60_0.94.sav', 
             'model_20_80_80_0.95.sav', 'model_40_100_60_0.94.sav']
model = pickle.load(open(os.path.join(sys.path[0], 'ANNmodel/'+modelFile[0]), 'rb'))

def denormVar(normalisedVar, varMin, varMax):
    denormalisedVar = (normalisedVar*(varMax - varMin)) + varMin
    return denormalisedVar

def readFile(filePath):
    caseList = []
    with open(filePath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')
        for case in readCSV:
            caseList.append(case)
    caseArray = np.asarray(caseList)
    Sigma, LambdaVar, Eta, Thickness, Spacing = caseArray[1:].T
    Sigma = np.asarray(map(float, Sigma))
    LambdaVar = np.asarray(map(float, LambdaVar))
    Eta = np.asarray(map(float, Eta))
    Thickness = np.asarray(map(float, Thickness))
    Spacing = np.asarray(map(float, Spacing))
    return Sigma, LambdaVar, Eta, Thickness, Spacing

def get_noise(yDim, xDim, white_noise_func_val):
    if white_noise_func_val == "Uniform":
        return rnd.random(size=(yDim, xDim))
    elif white_noise_func_val == "Beta":
        return rnd.beta(2, 2, size=(yDim, xDim))
    elif white_noise_func_val == "Normal":
        return rnd.normal(size=(yDim, xDim))
    elif white_noise_func_val == "Exponential":
        return rnd.exponential(size=(yDim, xDim))
    else:
        return rnd.laplace(size=(yDim, xDim))
    
def env_function(X, Y, env_func_val, sigma_val, m_val):
    if env_func_val == "Super Gaussian":
        return np.exp(-1 * ((X**2 / (sigma_val)**2) + (Y**2 / (sigma_val)**2))**m_val)
    else:# white_noise_func_val == "Cauchy":
        R = np.sqrt(X**2 + Y**2)
        return (1.0 / (np.pi * sigma_val)) * (sigma_val**2 / ((R/m_val)**2) + sigma_val**2)**m_val

def trabModel(env_func_val, m_val, white_noise_func_val, seed_noise_val, q0_val, r0_val, vis_val):
    if seed_noise_val:
        np.random.seed(100)
    paramsVal = []
    paramsName = []
    dim = 512
    yDim, xDim = dim, dim
    
    Sigma, LambdaVar, Eta, Thickness, Spacing = readFile(os.path.join(sys.path[0], 'data/trabData.csv'))
    
    thMin, thMax = np.min(Thickness), np.max(Thickness)
    xClickNorm = (xClick - thMin) / (thMax - thMin)
    spMin, spMax = np.min(Spacing), np.max(Spacing)
    yClickNorm = (yClick - spMin) / (spMax - spMin)
    
    predInput = np.array([xClickNorm, yClickNorm]).reshape(-1,1).T
    predVals = model.predict(predInput)
    sigmaPred, lambdaPred, etaPred = predVals[0][0], predVals[0][1], predVals[0][2]
    
    sigma_val = denormVar(sigmaPred, np.min(Sigma), np.max(Sigma))
    lamb_val = denormVar(lambdaPred, np.min(LambdaVar), np.max(LambdaVar))
    eta_val = denormVar(etaPred, np.min(Eta), np.max(Eta))
    
    paramsName.append(env_func_val)
    paramsVal.append(0)
    paramsName.append('m power')
    paramsVal.append(m_val)
    paramsName.append(white_noise_func_val)
    paramsVal.append(0)
    paramsName.append('Dim')
    paramsVal.append(dim)
    paramsName.append('Sigma')
    paramsVal.append(sigma_val)
    
    """ (a) create a map of pure white noise """
    pure_white_noise_map = get_noise(yDim, xDim, white_noise_func_val)

    """ (b) create a simple Lorentzian function, with a given standard deviation, and a given positional offset """
    x = np.linspace(-xDim, xDim, xDim)
    y = np.linspace(-yDim, yDim, yDim)
    [X,Y] = np.meshgrid(x,y)
    func_envelope = env_function(X, Y, env_func_val, sigma_val, m_val)
        
    trabOut_Model = np.zeros((yDim, xDim), np.uint8)
    pure_white_noise_map = pure_white_noise_map + (0.1 * get_noise(yDim, xDim, white_noise_func_val))
    enveloped_white_noise = func_envelope * pure_white_noise_map
    enveloped_white_noise_shifted = np.fft.fftshift(enveloped_white_noise)
    
    val = np.fft.fft2(enveloped_white_noise_shifted)
    val = val / np.abs(val)
    q0 = q0_val
    r0 = r0_val
    Eta = eta_val
    lambda_val = lamb_val
    
    paramsName.append('lambda_val')
    paramsVal.append(lamb_val)
    paramsName.append('q0')
    paramsVal.append(q0)
    paramsName.append('r0')
    paramsVal.append(r0)
    paramsName.append('Eta')
    paramsVal.append(Eta)
    
    r = ((X+q0)**2 + (Y+r0)**2)**(0.5)        
    spat = np.exp(-1j*lambda_val*r)
    if vis_val == 1:
        trab_Model = np.abs(1+spat)**(2*Eta)
    elif vis_val == 2:
        trab_Model = np.abs(val+1)**(2*Eta)
    else:
        trab_Model = np.abs(val+spat)**(2*Eta)
    
    """ Normalisation [0,1] """
    trab_Model = (trab_Model- np.min(trab_Model)) / (np.max(trab_Model) - np.min(trab_Model))
    
    thr_val = 0.5
    bools = trab_Model >= thr_val
    trabOut_Model[bools] = 255
    bools = trab_Model < thr_val
    trabOut_Model[bools] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    trabOut_Model = cv2.morphologyEx(trabOut_Model, cv2.MORPH_OPEN, kernel, iterations=1)
                  
    return trabOut_Model, paramsName, paramsVal

def save():
    fileName = 'trab_' + strftime('%y-%m-%d_%H-%M-%S', localtime())
    tf.imsave(base_path + fileName + '.tif', trabOutTmp)            
    
    print('Writing file %s'%fileName)
    with open(base_path + fileName + '.txt', 'w') as f:
        for idx, item in enumerate(paramsName,0):
            f.write("%s "%item)
            f.write("%2.2f\n"%paramsVal[idx])

def leftClick(event):
    global xClick, yClick
    if event.inaxes is None:
        print('Clicked ouside axes bounds but inside plot window')
    else:
        xClick, yClick = event.inaxes.transData.inverted().transform((event.x, event.y))
        print("Mouse click position: (%.1f %.1f)" %(xClick, yClick))
    return
  
def update():
    global sigmaClick, lambdaClick, etaClick
    global trabOutTmp, trabOut_Model, paramsName, paramsVal
    env_func_val = env_func.get()
    m_val = int(m.get())
    white_noise_func_val = white_noise_func.get()
    seed_noise_val = seed_noise.get()
    q0_val = q0.get()
    r0_val = r0.get()
    vis_val = vis.get()
    trabOutTmp, paramsName, paramsVal = trabModel(env_func_val, m_val, white_noise_func_val, 
                                                 seed_noise_val, q0_val, r0_val, vis_val)
    trabOut_Model = ImageTk.PhotoImage(Image.fromarray(trabOutTmp.astype('uint8')))
    canvasTrab.itemconfigure(imageTrab, image=trabOut_Model)
    ax.plot(xClick,yClick, 'x')
    fig.canvas.draw_idle()
    return

if __name__ == "__main__":    
    slider_length = 300
    save_val = 0
    
    window = Tk()
    window.title("Trabecular bone model")
    window.geometry('1800x600')
    
    """ Enveloping function definition """
    labelText=StringVar()
    labelText.set("Env. function")
    labelDir=Label(window, textvariable=labelText)
    labelDir.grid(column=0, row=0, sticky=E)
    
    env_func = ttk.Combobox(window, justify=CENTER)
    env_func['values']= ("Super Gaussian", "Cauchy")
    env_func.current(0) #set the selected item
    env_func.grid(column=1, row=0, columnspan=2, sticky=N, pady=2)  
    
    """ m power for defined function """
    labelText=StringVar()
    labelText.set("m")
    labelDir=Label(window, textvariable=labelText)
    labelDir.grid(column=19, row=0, sticky=E)
    
    mTmp = IntVar()
    mTmp.set(1) 
    m = Spinbox(window, from_=0, to=10, width=2, textvariable=mTmp)
    m.grid(column=20,row=0, pady=2)
        
    """ White noise function definition """
    labelText=StringVar()
    labelText.set("White noise")
    labelDir=Label(window, textvariable=labelText)
    labelDir.grid(column=0, row=1, sticky=E)
    
    white_noise_func = ttk.Combobox(window, justify=CENTER)
    white_noise_func['values']= ("Uniform", "Beta", "Normal", "Exponential", "Laplace")
    white_noise_func.current(0)
    white_noise_func.grid(column=1, row=1, sticky=W, pady=2)
    
    """ Check button """    
    seed_noise = IntVar()
    Checkbutton(window, text="Seed noise", onvalue=1, offvalue=0, variable=seed_noise).grid(column=20, row=1, sticky=W)    
    
    """ q0 for spatial function """
    labelText=StringVar()
    labelText.set("q_0")
    labelDir=Label(window, textvariable=labelText)
    labelDir.grid(column=0, row=3, sticky=E)
    
    q0 = Scale(window, from_=-250, to=250, tickinterval=100, length=slider_length, orient=HORIZONTAL)
    q0.grid(column=1, row=3, columnspan=20, pady=2, sticky=W)
    q0.set(0)
    
    """ r0 for spatial function """
    labelText=StringVar()
    labelText.set("r_0")
    labelDir=Label(window, textvariable=labelText)
    labelDir.grid(column=0, row=4, sticky=E)
    
    r0 = Scale(window, from_=-250, to=250, tickinterval=100, length=slider_length, orient=HORIZONTAL)
    r0.grid(column=1, row=4, columnspan=20, pady=2, sticky=W)
    r0.set(0)
            
    """ Image canvas grid """
    Sigma, LambdaVar, Eta, Thickness, Spacing = readFile(os.path.join(sys.path[0], 'data/trabData.csv'))
        
    fig = Figure(figsize=(4, 2))
    ax = fig.add_subplot(111)
    ax.plot(Thickness, Spacing, 'k.')
    ax.set_ylabel('Tb.Sp')
    ax.set_xlabel('Tb.Th')
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 60])
    
    canvasGrid = FigureCanvasTkAgg(fig, master=window)
    plot_widget = canvasGrid.get_tk_widget()
    plot_widget.grid(column=22, row=0, columnspan=3, rowspan=20, padx=5, sticky=W+E+N+S)
    fig.canvas.callbacks.connect('button_press_event', leftClick)

    """ Image canvas trab """
    canvasTrab = Canvas(window, width=512, height=512)
    canvasTrab.grid(column=25, row=0, columnspan=3, rowspan=20, padx=5, sticky=W+E+N+S)
    trabOut_Model = np.zeros((512,512))
    trabOut_Model =  ImageTk.PhotoImage(image=Image.fromarray(trabOut_Model))
    imageTrab = canvasTrab.create_image(0,0, anchor=NW, image=trabOut_Model)
     
    """ Radio buttons """
    vis = IntVar()
    vis.set(3)
    Label(window, text="""Choose a visualisation:""", justify=LEFT).grid(column=22, row=28, columnspan=3, sticky=N)
    Radiobutton(window, text="Spatial", variable=vis, value=1).grid(column=22, row=29, sticky=N)
    Radiobutton(window, text="Trabecular", variable=vis, value=2).grid(column=23, row=29, sticky=N)
    Radiobutton(window, text="Both", variable=vis, value=3).grid(column=24, row=29, sticky=N)
    
    """ Update box """
    button = Button(window, text="Update", command=update)
    button.grid(column=22, row=30, pady=10)
    
    """ Save box """
    button = Button(window, text="Save", command=save)
    button.grid(column=24, row=30, pady=10)
    
    window.mainloop()
    
    print("Done")
