# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, join
import Tkinter as tk, Tkconstants, tkFileDialog, tkMessageBox
from Tkinter import *
import Image, ImageTk
from recognition import *

class RvGui(tk.Frame):

  def __init__(self, root):

    tk.Frame.__init__(self, root)

    # display options
    opt1 = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}
    opt2 = {'fill': Tkconstants.BOTH, 'pady': 30}

    # definition of widgets for classifier training
    topFrame = Frame(root)
    topFrame.pack(opt2)    
    self.dir = StringVar()    
    tk.Label(topFrame, text="Učenje klasifikatora").pack()
    tk.Entry(topFrame, state='disabled', textvariable=self.dir).pack(**opt1)
    tk.Button(topFrame, text='Odaberi direktorij', command=self.askdirectory).pack(**opt1)
    self.buttonLearn = tk.Button(topFrame, text='Provedi učenje', state='disabled', command=self.learn)
    self.buttonLearn.pack(**opt1)

    # definition of widgets for prediction
    bottomFrame = Frame(root)
    bottomFrame.pack(opt1)    
    self.pic = StringVar()
    tk.Label(bottomFrame, text="Klasifikacija").pack()
    tk.Entry(bottomFrame, state='disabled', textvariable=self.pic).pack(**opt1)
    self.buttonChooseImage = tk.Button(bottomFrame, text='Odaberi sliku', state='disabled', command=self.askopenfilename)
    self.buttonChooseImage.pack(**opt1)
    self.buttonClassify = tk.Button(bottomFrame, text='Klasificiraj sliku', state='disabled', command=self.classify)
    self.buttonClassify.pack(**opt1)

    # directory options used in self.askdirectory
    self.dir_opt = options = {}
    options['initialdir'] = ''
    options['mustexist'] = False
    options['parent'] = root
    options['title'] = 'Odabir direktorija'

    # file options used in self.askopenfilename
    self.file_opt = options = {}
    options['defaultextension'] = '.pgm'
    options['initialfile'] = ''
    options['filetypes'] = [('All files', '.*'), ('Image files', '.pgm')]
    options['parent'] = root
    options['title'] = 'Odabir slike'


  def askdirectory(self):
    self.dir.set(tkFileDialog.askdirectory(**self.dir_opt))
    self.files = [ f for f in listdir(self.dir.get()) if isfile(join(self.dir.get(),f)) ]
    
    if len(self.dir.get()) > 1:
      self.buttonLearn['state'] = 'normal'
    else:
      self.buttonLearn['state'] = 'disabled'
            
    
  def learn(self):
    self.recognition = RecognitionArh2(unicode(self.dir.get(),"utf-8"))
    self.buttonChooseImage['state'] = 'normal'
    tkMessageBox.showinfo("Učenje","Učenje klasifikatora završeno")
    
    
  def askopenfilename(self):
    self.pic.set(tkFileDialog.askopenfilename(**self.file_opt))
    if len(self.pic.get()) > 1:
      self.buttonClassify['state'] = 'normal'
    else:
      self.buttonClassify['state'] = 'disabled'
      

  def classify(self):
    simils = self.recognition.get_similarities_for(self.pic.get())
    pos, sim, fold = max(simils, key = lambda x : x[1])
    self.new_window("Najsličnija slika", self.dir.get() + "/" + self.files[pos])

    
  def new_window(self, title, filename):
    window = tk.Toplevel(self)
    window.title("RV - " + title)
    window.geometry('300x300+200+200')    
    img = ImageTk.PhotoImage(Image.open(filename))
    panel = Label(window, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    window.mainloop()
    

if __name__=='__main__':
  root = tk.Tk()
  root.title('RV - Prepoznavanje lica')
  root.geometry('300x350+200+200')
  RvGui(root).pack()
  root.mainloop()
