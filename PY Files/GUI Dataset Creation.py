from tkinter import *
import tkinter.ttk as tpx
from PIL import Image, ImageTk, ImageOps
import os

CharacterSeg=[]
FileName=[]
path=os.getcwd()+'/Desktop/Dataset/Words/'
filename= os.listdir(path)
imgcounter=0
for name in filename:
    FileName.append(name)
r = Tk() 
r.configure(background='white')
r.geometry("900x350")
r.title('Dataset Creator')

def nextImage():
    newfile=open(globals()['path']+str(FileName[globals()['imgcounter']]).split('.')[0]+'.txt',"w+")
    newfile.write(str(globals()['CharacterSeg']))
    newfile.close()
    globals()['CharacterSeg']=[]
    globals()['imgcounter']+=1
    print(FileName[globals()['imgcounter']])
    imo=Image.open(globals()['path']+FileName[globals()['imgcounter']])
    width=imo.size[0]
    height=imo.size[1]
    imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
    imglabel.configure(image=imgtk,height=height,width=width)
    imglabel.image = imgtk
    imglabel.grid(column=0, row=0,padx=0)

nextbtn = Button(r,text="Next Image",command=nextImage)
nextbtn.grid(column=0, row=1,padx=5, pady=5,sticky=W)

imo=Image.open(path+FileName[imgcounter])
width=imo.size[0]
height=imo.size[1]
imglabel = Label(r,height=height,width=width)
imglabel.configure(background='black')

imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
imglabel.configure(image=imgtk)
imglabel.image = imgtk
imglabel.grid(column=0, row=0,padx=0)

def callback(event):
    CharacterSeg.append(event.x)
    print(f"X: {event.x} Y: {event.y}")
    print(f"Total Coordinates: {CharacterSeg}")

imglabel.bind('<Button-1>',callback)
def motion(event):
    # print(f"X: {event.x} Y: {event.y}")
    imo=Image.open(globals()['path']+FileName[globals()['imgcounter']])
    x=event.x
    y=event.y
    width=imo.size[0]
    height=imo.size[1]
    initialx=x-5
    initialy=y-5
    try:
        for i in range(7):
            for j in range(7):
                imo.putpixel((initialx+j,initialy+i),(255,0,0))
    except:
        pass

    imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
    imglabel.configure(image=imgtk,height=height,width=width)
    imglabel.image = imgtk
    imglabel.grid(column=0, row=0,padx=0)
    
imglabel.bind('<Motion>',motion)
    
r.mainloop()

