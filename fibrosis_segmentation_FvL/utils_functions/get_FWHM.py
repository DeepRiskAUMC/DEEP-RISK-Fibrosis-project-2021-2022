from cmath import exp
import os
import csv
from tkinter import *
import numpy as np
import tkinter.messagebox
from turtle import position
from PIL import Image, ImageTk, ImageEnhance
from utils_functions import get_data_paths
import SimpleITK as sitk

data_path = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing'
test_folder = os.path.join(data_path, 'test')
test_path = os.path.join(test_folder, 'LGE_niftis')
all_LGE_paths = get_data_paths(test_path)

positions = {}
all_LGE_paths = [all_LGE_paths[0]]

for nifti_path in all_LGE_paths:
    pat_id = nifti_path.split('\\')[-1].split('_')[0]
    positions[pat_id] = {}
    pat_dict = positions[pat_id]
    LGE_img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path)).squeeze()
    LGE_img = LGE_img - np.min(LGE_img)
    LGE_img = (LGE_img / np.max(LGE_img) * 255).astype(int)
    slices_count = LGE_img.shape[0]
    for i in range(slices_count):
        img_array = LGE_img[i]
        img = Image.fromarray(img_array)
        clicked_positions = []

        # img = Image.open("output\\AUMC2D\\myocard\\version_0\\train\\prediction_DRAUMC0137_slice6.png")
        img_h, img_w = img.height, img.width
        new_img_h, new_img_w = img_h*3, img_w*3
        img = img.resize((new_img_w, new_img_h), Image.BILINEAR)
        img = img.convert('L')

        mGui = Tk()
        mGui.title('Klik op een witte pixel in alle hyper aangekleurde gebieden')
        mGui.geometry(f'{new_img_w+150}x{new_img_h}')
        mGui.resizable(0, 0) #Disable Resizeability
        photoFrame = Frame(mGui, bg="orange", width=new_img_w, height=new_img_h)
        photoFrame.pack(side=LEFT)
        rightFrame = Frame(mGui, bg="white", width=150, height=new_img_h)
        rightFrame.pack(side=BOTTOM, fill=Y, expand=True)
        buttonsFrame = Frame(rightFrame, bg="yellow", width=150, height=int(new_img_h/4))
        buttonsFrame.pack(side=BOTTOM)
        bsliderFrame = Frame(rightFrame, bg="white", width=150, height=int(new_img_h/8))
        bsliderFrame.pack(side=BOTTOM, fill=Y)
        csliderFrame = Frame(rightFrame, bg="white", width=150, height=int(new_img_h/8))
        csliderFrame.pack(side=BOTTOM, fill=Y)
        titleFrame = Frame(rightFrame, bg="white", width=150, height=int(new_img_h/4))
        titleFrame.pack(side=TOP)
        positionsFrame = Frame(rightFrame, bg="white", width=150, height=int(new_img_h/4))
        positionsFrame.pack(side=TOP, fill=Y, expand=True)
        # label2 = Label(photoFrame)

        patient_label = Label(titleFrame, text=f'Patient {pat_id}')
        patient_label.pack(side = TOP)
        slice_label = Label(titleFrame, text=f'slice {i}')
        slice_label.pack(side = TOP)

        clicked_label = Label(positionsFrame, text="")
        clicked_label.pack(side = TOP)

        # Create a Label Widget to display the text or Image

        photo_img = ImageTk.PhotoImage(img)
        photo_label = Label(photoFrame, image = photo_img)
        photo_label.pack(side = TOP)

        def adjContrast(new_contrast):
            enhancer = ImageEnhance.Contrast(img)
            factor = float(new_contrast)
            adjusted_img = enhancer.enhance(factor)
            adjusted_photo_img = ImageTk.PhotoImage(adjusted_img)
            photo_label.config(image=adjusted_photo_img)
            photo_label.image = adjusted_photo_img

        def adjBright(new_brightness):
            enhancer = ImageEnhance.Brightness(img)
            factor = float(new_brightness)
            adjusted_img = enhancer.enhance(factor)
            adjusted_photo_img = ImageTk.PhotoImage(adjusted_img)
            photo_label.config(image=adjusted_photo_img)
            photo_label.image = adjusted_photo_img

        def removeClick():
            global clicked_positions
            clicked_positions = clicked_positions[:-1]
            string_positions = get_string(clicked_positions)
            print(clicked_positions)
            clicked_label.configure(text=string_positions)

        def nextImage():
            pat_dict[i] = clicked_positions
            mGui.destroy()
        
        # sliderScale = Scale(sliderFrame, variable=IntVar())
        variable = DoubleVar()
        brightnessSlider = Scale(bsliderFrame, from_=0.0, to=2.0, orient=HORIZONTAL, command=adjBright, digits=2, resolution=0.1, length=200 ,width=10, sliderlength=15)
        brightnessSlider.set(1.0)
        brightnessSlider.pack(side = BOTTOM, anchor=CENTER)
        bsliderLabel = Label(bsliderFrame, text="Adjust brightness")
        bsliderLabel.pack(side = BOTTOM)
        contractSlider = Scale(csliderFrame, from_=0.0, to=2.0, orient=HORIZONTAL, command=adjContrast, digits=2, resolution=0.1, length=200 ,width=10, sliderlength=15)
        contractSlider.set(1.0)
        contractSlider.pack(side = BOTTOM, anchor=CENTER)
        csliderLabel = Label(csliderFrame, text="Adjust contrast")
        csliderLabel.pack(side = BOTTOM)

        #Create Buttons for All the Possible Filters
        done_btn = Button(buttonsFrame, text="Done", command=nextImage)
        done_btn.pack(side = BOTTOM, pady = 2)

        confirm_btn = Button(buttonsFrame, text="Remove last click")
        confirm_btn.pack(side = BOTTOM, pady = 2)

        def get_string(positions):
            string_pos = ''
            for position in positions:
                string_pos = string_pos + str(position) + '\n'
            return string_pos

        def getorigin(eventorigin):
            if eventorigin.widget == done_btn:
                nextImage()
            elif eventorigin.widget == confirm_btn:
                removeClick()
            elif eventorigin.widget == photo_label:
                global x,y
                x = eventorigin.x
                y = eventorigin.y
                clicked_positions.append((int(y/3),int(x/3)))
                string_positions = get_string(clicked_positions)
                clicked_label.configure(text=string_positions)
                mGui.bind("<Button 1>",getorigin)
        mGui.bind("<Button 1>",getorigin)

        mGui.mainloop()

with open(os.path.join('L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\test', 'FWHM_locations.csv'), 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    for pat_id in positions:
        slices_dict = positions[pat_id]
        for slice_number in slices_dict:
            row  = [pat_id, str(slice_number)]
            slice_positions = slices_dict[slice_number]
            for position in slice_positions:
                row.append(str(position))
            writer.writerow(row)