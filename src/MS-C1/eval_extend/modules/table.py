from tkinter import *
import filemod
import webmod



d = {}
f = open('1mlist.tsv',encoding='utf-8')
for i in f:
	if '@en' in i:
		mid = i.split('\t')[0]
		name = i.strip().split('\t')[1].replace('@en','').replace('\"','')
		if not mid in d:
			d[mid] = name
f.close()

d100k = {}
f = open('listid.txt')
for i in f:
	i = i.strip()
	if not i in d100k:
		d100k[i]=0
f.close()


class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling
    """
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set,height=800)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            # print('size',size)
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

root = Tk()
rightimg = PhotoImage(file='./imgassets/right.png')
wrongimg = PhotoImage(file='./imgassets/wrong.png')

def generateLine(data):
	frame = Frame(root)
	frame.pack(padx=2,pady=2)
	assert len(data)==5
	for i in range(5):
		if i==4:
			lb = Label(frame,font=('Times',15,'bold'),bg='#e5e5e5',text=data[i],width=7)
		elif i==0:
			lb = Label(frame,font=('Times',15,'bold'),bg='#e5e5e5',text=data[i],width=10)
		else:
			lb = Label(frame,font=('Times',15,'bold'),bg='#e5e5e5',text=data[i],width=20)
		lb.grid(row=0,column=i)

generateLine(['#','GroundTruth','Prediction','Score','T/F'])

tableframe = VerticalScrolledFrame(root)
tableframe.pack()
tableframe = tableframe.interior
tframe = Frame(tableframe)
tframe.pack()

textvar = StringVar()

def callback(txt):
	textvar.set(txt)

linenumber = 0

def generateLine2(data):
	global linenumber,tframe
	if linenumber%2==1:
		bgcolor = '#e5e5e5'
	else:
		bgcolor = '#ffffff'
	# if linenumber%100==0:
	# 	tframe = Frame(tableframe)
	# 	tframe.pack()
	frame = Frame(tframe,bg=bgcolor)
	assert len(data)==5
	Label(frame,text=data[0],font=(None, 10),bg=bgcolor,width=15,anchor=CENTER).grid(row=0,column=0)
	Button(frame,text=data[1],font=(None, 10),bg=bgcolor,bd=0,width=30,command=lambda: callback(data[1])).grid(row=0,column=1)
	Button(frame,text=data[2],font=(None, 10),bg=bgcolor,bd=0,width=30,command=lambda: callback(data[2])).grid(row=0,column=2)
	Label(frame,text=data[3],font=(None, 10),bg=bgcolor,width=30,anchor=CENTER).grid(row=0,column=3)
	if data[4]==1:
		Label(frame,font=(None, 10),image=rightimg,bg=bgcolor,width=30,anchor=CENTER).grid(row=0,column=4)
	else:
		Label(frame,font=(None, 10),image=wrongimg,bg=bgcolor,width=30,anchor=CENTER).grid(row=0,column=4)
	if data[1] in d100k:
		Label(frame,font=(None, 10),image=rightimg,bg=bgcolor,width=30,anchor=CENTER).grid(row=0,column=6)
	else:
		Label(frame,font=(None, 10),image=wrongimg,bg=bgcolor,width=30,anchor=CENTER).grid(row=0,column=6)
	frame.grid(row=linenumber,column=0,padx=2,pady=2)
	linenumber+=1

def generateTable(data):
	# print('data length:',len(data))
	# print(data[-1])
	for i in data:
		generateLine2(i)

def GoogleCallback():
	t = textvar.get()
	if t in d:
		webmod.open(d[t])
	else:
		tkMessageBox.showinfo( 'No key error', 'MID '+t+' does not exist in dictionary!')

def TrainCallback():
	t = textvar.get()
	path = 'I:\\database\\base\\'+t
	filemod.open(path)

def TestCallback():
	t = textvar.get()
	path = 'H:\\MS\\Development_Set\\set1\\aligned\\'+t+'.jpg'
	filemod.open(path)

def OriginalCallback():
	t = textvar.get()
	path = 'I:\\mscolor\\'+t
	filemod.open(path)

def getTable(data):
	fout = open('rightmid.txt','w')
	for i in data:
		if i[4]==1:
			fout.write(i[1]+'\t'+str(i[3])+'\n')
	fout.close()
	generateTable(data)
	innerFrame = Frame(root)
	innerFrame.pack()
	Label(innerFrame,font=("Helvetica", 13,'bold'),text='m.id:  ').grid(row=0,column=0)
	textEdit = Entry(innerFrame,font=("Helvetica", 13),textvariable=textvar).grid(row=0,column=1,padx=5)
	Button(innerFrame,font=("Helvetica", 13),text='Google search',command=GoogleCallback).grid(row=0,column=2,padx=5)
	Button(innerFrame,font=("Helvetica", 13),text='Training data',command=TrainCallback).grid(row=0,column=3,padx=5)
	Button(innerFrame,font=("Helvetica", 13),text='Original data',command=OriginalCallback).grid(row=0,column=4,padx=5)
	Button(innerFrame,font=("Helvetica", 13),text='Testing data',command=TestCallback).grid(row=0,column=5,padx=5)
	mainloop()
