import os
from tkinter import *
import tkinter
import arabic_reshaper
from bidi.algorithm import get_display
from tkinter import filedialog
from tkinter import messagebox
import time
from time import strftime
from PIL import Image, ImageTk, ImageEnhance,ImageFont
from persiantools.jdatetime import JalaliDate
import sqlite3
from embedding import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import sys
# sys.path.append('/home/sm/Documents/opencv411/opencv411_installed/lib/python3.8/dist-packages')
import cv2
import torchvision.ops as ops
import torchvision.models.resnet as resnet
from torchvision_model import RetinaFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, ArcFace, Facenet512, DlibResNet
from deepface.commons import functions, realtime, distance as dst
from datetime import datetime
import csv
from queue import Queue



quitt = "آیا میخواهید خارج شوید؟"
reshaped_txt = arabic_reshaper.reshape(quitt)
quitt = get_display(reshaped_txt)


def on_closing():
    if messagebox.askyesno("خروج", quitt):
        parent.destroy()
        # close database--------------------------------------------------
        conn.close()


# contact-----------------------------------------------------------------
message1 = " از طریق شماره زیر با ما در ارتباط باشید." \
           " ۰۲۶۳۴۷۶۰۰۹۴ "
reshaped_txt = arabic_reshaper.reshape(message1)
message1 = get_display(reshaped_txt)


def contact():
    messagebox._show(title="ارتباط با ما",
                     message=message1)


def convertToBinaryData(filename):
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def addnew():
    name = nameVar.get()
    family = lastnameVar.get()
    fname = fathernameVar.get()
    ID = IDVar.get()
    gender = genderVar.get()
    national = nationalVar.get()
    photo = imagepath.get()

    conn = sqlite3.connect('/home/bagheri/PycharmProjects/pythonProject4/testUI2.db')
    with conn:
        cursor = conn.cursor()

    cursor.execute(
        'CREATE TABLE IF NOT EXISTS personInfo_1 (Name CHAR(20) NOT NULL,family CHAR(30) NOT NULL,fname CHAR(20) NOT NULL,ID CHAR(10) PRIMARY KEY NOT NULL,gender CHAR(1) NOT NULL,national CHAR(1) NOT NULL, photo BLOB NOT NULL ,embedding CHAR(2622) NOT NULL)')
    image = convertToBinaryData(photo)
    embedding = main(photo)

    cursor.execute(
        'INSERT INTO personInfo_1 (Name,family,fname,ID,Gender,national,photo,embedding) VALUES(?,?,?,?,?,?,?,?)',
        (name, family, fname, ID, gender, national, image, embedding))
    if cursor.fetchone() is not None:
        print("Welcome")
    else:
        print("Login failed")

    conn.commit()


def embedding_query():
    embeddings = []
    connectionc = sqlite3.connect('/home/bagheri/PycharmProjects/pythonProject4/testUI2.db')
    cursor = connectionc.cursor()
    sql_fetch_blob_query = """SELECT DISTINCT embedding from personInfo_1 """
    cursor.execute(sql_fetch_blob_query)
    record = cursor.fetchall()
    for i in range(len(record)):
        embedding = np.frombuffer(record[i][0], np.float32)
        embeddings.append(embedding)
    return embeddings


def name_query():
    fnames = []
    lnames = []
    tab = [' ']
    names = []
    connectionc = sqlite3.connect('/home/bagheri/PycharmProjects/pythonProject4/testUI2.db')
    cursor = connectionc.cursor()
    name_selection = """SELECT name,family from personInfo_1 """
    cursor.execute(name_selection)
    record = cursor.fetchall()
    for row in record:
        fnames.append(row[0])
        lnames.append(row[1])

    for i in range(len(fnames)):
        name = fnames[i] + tab[0] + lnames[i]
        names.append(name)
    return names


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def face_distance(known_face_encodings, face_img_embedding):
    face_distance = []
    for faces in known_face_encodings:
        distance = dst.findCosineDistance(faces, face_img_embedding[0])
        face_distance.append(distance)

    return face_distance


def compare_faces(known_face_encodings, face_img_embedding, tolerance=0.23):
    compare_face = []
    for facedst in face_distance(known_face_encodings, face_img_embedding):
        if facedst <= tolerance:
            compare_face.append(True)
        else:
            compare_face.append(False)
    return compare_face


def detected_area(box):
    a = box[0].item()
    b = box[1].item()
    c = box[2].item()
    d = box[3].item()
    area = (d - b) * (c - a)
    return area


def juge_face(picked_landmark):
    indexs = range(len(picked_landmark))
    for index in indexs:
        landmark_i = picked_landmark[index]
        x = int(landmark_i[0].item())
        landmark_j = picked_landmark[index]
        y = int(landmark_j[1].item())
        left_eye = (x, y)

        landmark_i = picked_landmark[index]
        x = int(landmark_i[2].item())
        landmark_j = picked_landmark[index]
        y = int(landmark_j[3].item())
        right_eye = (x, y)

        landmark_i = picked_landmark[index]
        x = int(landmark_i[4].item())
        landmark_j = picked_landmark[index]
        y = int(landmark_j[5].item())
        nose = (x, y)

        landmark_i = picked_landmark[index]
        x = int(landmark_i[6].item())
        landmark_j = picked_landmark[index]
        y = int(landmark_j[7].item())
        mouth_left = (x, y)

        landmark_i = picked_landmark[index]
        x = int(landmark_i[8].item())
        landmark_j = picked_landmark[index]
        y = int(landmark_j[9].item())
        mouth_right = (x, y)

        right_eye_array = np.array(right_eye)
        left_eye_array = np.array(left_eye)
        nose_array = np.array(nose)
        mouth_right_array = np.array(mouth_right)
        mouth_left_array = np.array(mouth_left)

        vec_A = right_eye_array - nose_array
        vec_B = left_eye_array - nose_array
        vec_C = mouth_right_array - nose_array
        vec_D = mouth_left_array - nose_array
        dist_A = np.linalg.norm(vec_A)
        dist_B = np.linalg.norm(vec_B)
        dist_C = np.linalg.norm(vec_C)
        dist_D = np.linalg.norm(vec_D)

        high_rate = dist_A / dist_C
        width_rate = dist_C / dist_D

        high_ratio_variance = np.fabs(high_rate - 1.1)  # smaller is better
        width_ratio_variance = np.fabs(width_rate - 1)

        wide_dist = np.linalg.norm(right_eye_array - left_eye_array)
        high_dist = np.linalg.norm(right_eye_array - mouth_right_array)
        dist_rate = high_dist / wide_dist
        return dist_rate, width_ratio_variance


def known_face():
    known_face_encodings = embedding_query()
    known_face_names = name_query()
    return known_face_encodings, known_face_names


def matching(known_face_encodings, face_img_embedding, known_face_names):
    matches = compare_faces(known_face_encodings, face_img_embedding, 0.23)
    name = " "

    face_distances = face_distance(known_face_encodings, face_img_embedding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    return name


def unknown_face(face_img, face_model):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = np.expand_dims(face_img, axis=0)
    face_img_embedding = [(face_model.predict(face_img, verbose=0)[0, :])]
    return face_img_embedding


# Define function to show frame
def deteced_face():
    cv2image = get_frames()
    # small_img = cv2.resize(cv2image,(0,0),fx= 0.2,fy=0.2)
    picked_boxes, picked_landmarks, picked_scores = create_box(cv2image)
    area_list = []
    face_box = []
    log_time = []
    date = []
    face_list_save = []



    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, landmark, score in zip(boxes, picked_landmarks[j], picked_scores[j]):

                area = detected_area(box)
                picked_landmark = picked_landmarks[0]
                dist_rate, width_ratio_variance = juge_face(picked_landmark)
                if area > 204 and dist_rate < 1.4 and width_ratio_variance < 1:
                    face_img = cv2image[(int(box[1]) * 5):(int(box[3]) * 5), (int(box[0]) * 5):(int(box[2]) * 5)]
                    face_img = cv2.copyMakeBorder(face_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None,
                                                  value=(255, 255, 255))
                    area_list.append(area)
                    face_box.append(face_img)
                    time = datetime.now()
                    login_time = time.strftime("%H:%M:%S")
                    dates = datetime.today()
                    date_time = dates.strftime("%Y:%m:%d")
                    log_time.append(login_time)
                    date.append(date_time)
                    face_list_save.append(face_img)

    index = np.argsort(area_list)
    index = index[::-1]
    face_list1 = []
    for i in index[0:12]:
        face_list1.append(face_box[i])
    return face_list1 , date ,log_time ,face_list_save
def detection_save():
    image_path = "/home/jalali/algorithm/detected_face/"
    is_exist = os.path.exists(image_path)
    if is_exist == False:
        os.makedirs(image_path)

    face_list, date, log_time,face_list_save = deteced_face()
    detected = []
    for face in range(len(face_list)):
        image = Image.fromarray(np.uint8(face_list[face]))
        image = image.resize((105, 100), Image.BILINEAR)
        image = ImageTk.PhotoImage(image)
        detected.append(image)

    for j in range(len(face_list_save)):
        face_list_save[j] = cv2.cvtColor(face_list_save[j], cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path + "" + str(date[j]+log_time[j]) + ".jpeg", face_list_save[j])
    return detected

def show_frames():
    label1 = Label(parent)
    label2 = Label(parent)
    label3 = Label(parent)
    label4 = Label(parent)
    label5 = Label(parent)
    label6 = Label(parent)
    label7 = Label(parent)
    label8 = Label(parent)
    label9 = Label(parent)
    label10 = Label(parent)
    label11 = Label(parent)
    label12 = Label(parent)

    labels = [label1, label2, label3, label4, label5, label6, label7, label8, label9, label10, label11, label12]
    detected = detection_save()
    for face in range(len(detected)):
        if face == 0:
            label1.imgtk =detected[0]
            label1.configure(image=detected[0])
            label1.place(x=1000, y=90, relwidth=0.07)
            label1.after(1000, label1.destroy)
        elif face == 1:
            label2.imgtk = detected[1]
            label2.configure(image=detected[1])
            label2.place(x=1120, y=90, relwidth=0.07)
            label2.after(1000, label2.destroy)
        elif face == 2:
            label3.imgtk = detected[2]
            label3.configure(image=detected[2])
            label3.place(x=1240, y=90, relwidth=0.07)
            label3.after(1000, label3.destroy)
        elif face == 3:
            label4.imgtk = detected[3]
            label4.configure(image=detected[3])
            label4.place(x=1360, y=90, relwidth=0.07)
            label4.after(1000, label4.destroy)
        elif face == 4:
            label5.imgtk = detected[4]
            label5.configure(image=detected[4])
            label5.place(x=1000, y=210, relwidth=0.07)
            label5.after(1000, label5.destroy)
        elif face == 5:
            label6.imgtk = detected[5]
            label6.configure(image=detected[5])
            label6.place(x=1120, y=210, relwidth=0.07)
            label6.after(1000, label6.destroy)
        elif face == 6:
            label7.imgtk = detected[6]
            label7.configure(image=detected[6])
            label7.place(x=1240, y=210, relwidth=0.07)
            label7.after(1000, label7.destroy)
        elif face == 7:
            label8.imgtk = detected[7]
            label8.configure(image=detected[7])
            label8.place(x=1360, y=210, relwidth=0.07)
            label8.after(1000, label8.destroy)
        elif face == 8:
            label9.imgtk = detected[8]
            label9.configure(image=detected[8])
            label9.place(x=1000, y=330, relwidth=0.07)
            label9.after(1000, label9.destroy)
        elif face == 9:
            label10.imgtk = detected[9]
            label10.configure(image=detected[9])
            label10.place(x=1120, y=330, relwidth=0.07)
            label10.after(1000, label10.destroy)
        elif face == 10:
            label11.imgtk = detected[10]
            label11.configure(image=detected[10])
            label11.place(x=1240, y=330, relwidth=0.07)
            label11.after(1000, label11.destroy)
        elif face == 11:
            label12.imgtk = detected[11]
            label12.configure(image=detected[11])
            label12.place(x=1360, y=330, relwidth=0.07)
            label12.after(1000, label12.destroy)

    # ----------------------------------------------------------------

    label.after(1000, show_frames)
def recognition_save():
    face_list, date, log_time,face_list_save = deteced_face()
    detect = []
    name = []
    date_time1 = []
    loging_time = []

    for f in range(len(face_list[0:3])):
        face_img_embedding = unknown_face(face_list[f], face_model)
        match = matching(known_face_encodings, face_img_embedding, known_face_names)
        if match != " ":
            name.append(match)
            image = Image.fromarray(np.uint8(face_list[f]))
            image = image.resize((140, 160), Image.BILINEAR)
            image = ImageTk.PhotoImage(image)
            detect.append(image)

            time = datetime.now()
            login_time = time.strftime("%H:%M:%S")
            dates = datetime.today()
            date_time = dates.strftime("%Y:%m:%d")

            loging_time.append(login_time)
            date_time1.append(date_time)

            with open("/home/jalali/algorithm/detected_face/recognitionface.csv", "a+") as csvfile:
                writer = csv.writer(csvfile)
                for value in range(len(name)):
                    writer.writerow([name[value], date_time1[value], loging_time[value]])
    return detect,name


def recognition():


    detect, name_list = recognition_save()
    # print('name',name)
    # print('face', detect)
    label13 = Label(parent)
    label14 = Label(parent)
    label15 = Label(parent)
    label16 = Label(parent)
    label17 = Label(parent)
    label18 = Label(parent)

    for i in range(len(detect[0:3])):
        if len(detect) > 0:
            if i == 0:
                label13.imgtk = detect[0]
                label13.place(x=980, y=500, relwidth=0.1, relheight=0.2)
                label13.configure(image=detect[0])
                # label13.after(1000, label13.destroy)

                reshaped_txt = arabic_reshaper.reshape(name_list[0])
                name = get_display(reshaped_txt)
                if len(name) > 16:
                    label16 = Label(parent, text=name, font=('times', 14, ' bold '))
                else:
                    label16 = Label(parent, text=name, font=('times', 16, ' bold '))
                label16.pack(side=RIGHT)
                label16.place(x=1000, y=680)
                # label16.after(1000, label16.destroy)

            elif i  == 1:

                label14.imgtk = detect[1]
                label14.place(x=1160, y=500, relwidth=0.1, relheight=0.2)
                label14.configure(image=detect[1])
                # label14.after(1000, label14.destroy)

                reshaped_txt = arabic_reshaper.reshape(name_list[1])
                name1= get_display(reshaped_txt)
                if len(name) > 16:
                    label17 = Label(parent, text=name1, font=('times', 14, ' bold '))
                else:
                    label17 = Label(parent, text=name1, font=('times', 16, ' bold '))

                label17.pack(side=RIGHT)
                label17.place(x=1200, y=680)
                # label17.after(1000, label17.destroy)

            elif i  == 2:

                label15.imgtk = detect[2]
                label15.place(x=1340, y=500, relwidth=0.1, relheight=0.2)
                label15.configure(image=detect[2])
                # label15.after(1000, label15.destroy)

                reshaped_txt = arabic_reshaper.reshape(name_list[2])
                name2 = get_display(reshaped_txt)
                if len(name) > 16:
                    label18 = Label(parent, text=name2, font=('times', 14, ' bold '))
                else:
                    label18 = Label(parent, text=name2, font=('times', 16, ' bold '))
                label18.pack(side=RIGHT)
                label18.place(x=1380, y=680)
                # label18.after(1000, label18.destroy)
    label.after(500, recognition)

def get_frames():
    try:
        # cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        cv2image =cap
        return cv2image
    except:

        msg = "ارتباط با دوربین برقرار نمیباشد"
        reshaped_txt = arabic_reshaper.reshape(msg)
        msg = get_display(reshaped_txt)
        message = messagebox.showinfo("هشدار", msg)

def show_camera():
    # Get the latest frame and convert into Image
    # cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)

    cv2image = get_frames()
    # small_img = cv2.resize(cv2image, (0, 0), fx=0.5, fy=0.5)


    picked_boxes, picked_landmarks, picked_scores = create_box(cv2image)

    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, landmark, score in zip(boxes, picked_landmarks[j], picked_scores[j]):

                area = detected_area(box)
                picked_landmark = picked_landmarks[0]
                dist_rate, width_ratio_variance = juge_face(picked_landmark)
                print("area", area)
                # if area > 15 and dist_rate < 1.4 and width_ratio_variance < 1:
                cv2.rectangle(cv2image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255),
                              thickness=2)


    # ------------------------------
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.place(x=135, y=80)
    # Repeat after an interval to capture continiously
    label.after(4, show_camera)



def create_box(image):
    # Read video
    while (True):
        img = image
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        input_img = img.unsqueeze(0).float().cuda()

        picked_boxes, picked_landmarks, picked_scores = get_detections(input_img, RetinaFace, score_threshold=0.5,
                                                                       iou_threshold=0.3)
        return picked_boxes, picked_landmarks, picked_scores


def get_detections(img_batch, model, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        classifications, bboxes, landmarks = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        picked_scores = []

        for i in range(batch_size):
            classification = torch.exp(classifications[i, :, :])
            bbox = bboxes[i, :, :]
            landmark = landmarks[i, :, :]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax == 0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice

            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_landmarks.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]
            landmark = landmark[positive_indices]

            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_landmarks = landmark[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_landmarks.append(keep_landmarks)
            picked_scores.append(keep_scores)

        return picked_boxes, picked_landmarks, picked_scores


def create_retinaface(return_layers, backbone_name='resnet50', anchors_num=3, pretrained=True):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
    # freeze layer1
    for name, parameter in backbone.named_parameters():
        if name == 'conv1.weight':
            parameter.requires_grad_(False)

    model = RetinaFace(backbone, return_layers, anchor_nums=3)

    return model


def time():
    string = strftime('%H:%M:%S %p')
    lbl.config(text=string)
    lbl.after(1000, time)


def selection():
    pass






def add():
    # second window------------------------------------------------
    child = Toplevel(parent)
    child.geometry("1120x500")
    child.resizable(0, 0)
    child.configure(background='#97c1e8')
    child.title("اضافه کردن شخص جدید")

    def only_numbers(char):
        return char.isdigit()

    def only_char(char):
        return char.isalpha()

    validation = child.register(only_numbers)
    validation1 = child.register(only_char)

    def time():
        string = strftime('%H:%M:%S %p')
        lbl.config(text=string)
        lbl.after(1000, time)

    lbl = Label(child, font=('calibri', 15, 'bold'),
                background='#5454FF',
                foreground='white')

    # Placing clock at the centre
    # of the tkinter window
    lbl.pack()
    lbl.place(x=890, y=40, relwidth=0.2)
    time()

    # -----------------------------------------------------------------

    info = "مشخصات فردی"
    reshaped_txt = arabic_reshaper.reshape(info)
    info = get_display(reshaped_txt)
    message3 = tkinter.Label(child, text=info, fg="white", bg="#0a4089", width=10, height=1,
                             font=('times', 19, ' bold '))
    message3.place(x=3, y=3, relwidth=1)

    # -----------------------------------------------------------------
    name = "نام"
    reshaped_txt = arabic_reshaper.reshape(name)
    name = get_display(reshaped_txt)
    L1 = Label(child, text=name, font=('times', 16, ' bold '))
    L1.pack(side=RIGHT)
    L1.place(x=1000, y=160, relwidth=0.1)
    E1 = Entry(child, bd=5, textvar=nameVar, justify='right', validate="key", validatecommand=(validation1, '%S'))
    E1.pack(side=LEFT)
    E1.place(x=750, y=150, relwidth=0.2, relheight=0.08)
    E1.focus_set()

    family = "نام خانوادگی"
    reshaped_txt = arabic_reshaper.reshape(family)
    family = get_display(reshaped_txt)
    L2 = Label(child, text=family, font=('times', 16, ' bold '))
    L2.pack(side=LEFT)
    L2.place(x=580, y=160, relwidth=0.1)
    E2 = Entry(child, bd=5, textvar=lastnameVar, justify='right', validate="key",
               validatecommand=(validation1, '%S'))
    E2.pack(side=LEFT)
    E2.place(x=330, y=150, relwidth=0.2, relheight=0.08)

    fname = "نام پدر"
    reshaped_txt = arabic_reshaper.reshape(fname)
    fname = get_display(reshaped_txt)
    L3 = Label(child, text=fname, font=('times', 16, ' bold '))
    L3.pack(side=LEFT)
    L3.place(x=1000, y=210, relwidth=0.1)
    E3 = Entry(child, bd=5, textvar=fathernameVar, justify='right', validate="key",
               validatecommand=(validation1, '%S'))
    E3.pack(side=LEFT)
    E3.place(x=750, y=200, relwidth=0.2, relheight=0.08)

    cart = "کد ملی"
    reshaped_txt = arabic_reshaper.reshape(cart)
    cart = get_display(reshaped_txt)
    L4 = Label(child, text=cart, font=('times', 16, ' bold '))
    L4.pack(side=LEFT)
    L4.place(x=580, y=210, relwidth=0.1)
    E4 = Entry(child, bd=5, validate="key", validatecommand=(validation, '%S'), textvar=IDVar)
    E4.pack(side=LEFT)
    E4.place(x=330, y=200, relwidth=0.2, relheight=0.08)

    sex = "جنسیت"
    reshaped_txt = arabic_reshaper.reshape(sex)
    sex = get_display(reshaped_txt)
    L5 = Label(child, text=sex, font=('times', 16, ' bold '))
    L5.pack(side=LEFT)
    L5.place(x=1000, y=260, relwidth=0.1)

    name = "زن"
    reshaped_txt = arabic_reshaper.reshape(name)
    name = get_display(reshaped_txt)
    r1 = Radiobutton(child, text=name, variable=genderVar, value=1, font=('times', 14, ' bold '))
    r1.pack(anchor=W)
    r1.place(x=750, y=250, relwidth=0.2, relheight=0.08)
    name = "مرد"
    reshaped_txt = arabic_reshaper.reshape(name)
    name = get_display(reshaped_txt)
    r2 = Radiobutton(child, text=name, variable=genderVar, value=2, font=('times', 14, ' bold '))
    r2.pack(anchor=W)
    r2.place(x=750, y=280, relwidth=0.2, relheight=0.08)

    national = "ملیت"
    reshaped_txt = arabic_reshaper.reshape(national)
    national = get_display(reshaped_txt)
    L1 = Label(child, text=national, font=('times', 16, ' bold '))
    L1.pack(side=LEFT)
    L1.place(x=580, y=250, relwidth=0.1)

    name = "ایرانی"
    reshaped_txt = arabic_reshaper.reshape(name)
    name = get_display(reshaped_txt)
    r3 = Radiobutton(child, text=name, variable=nationalVar, value=1, command=selection,
                     font=('times', 14, ' bold '))
    r3.pack(anchor=N)
    r3.place(x=330, y=250, relwidth=0.2, relheight=0.08)
    name = "غیر ایرانی"
    reshaped_txt = arabic_reshaper.reshape(name)
    name = get_display(reshaped_txt)
    r4 = Radiobutton(child, text=name, variable=nationalVar, value=2, command=selection,
                     font=('times', 14, ' bold '))
    r4.pack(anchor=N)
    r4.place(x=330, y=280, relwidth=0.2, relheight=0.08)

    canvas_child1 = Canvas(child, width=250, height=250)
    canvas_child1.pack()
    canvas_child1.place(x=15, y=150, relwidth=0.2)

    canvas_child = Canvas(canvas_child1, width=250, height=250)
    canvas_child.pack()
    canvas_child.place(x=0, y=0, relwidth=1)

    # --------------------------------------------------------------------
    def save():

        text_name = nameVar.get()
        text_lastname = lastnameVar.get()
        text_fathername = fathernameVar.get()
        text_ID = IDVar.get()
        val_gender = genderVar.get()
        val_national = nationalVar.get()
        text_path = imagepath.get()

        if len(text_name) == 0 or len(text_lastname) == 0 or len(text_fathername) == 0 or len(text_ID) == 0:
            msg = " فیلد خالی را پر کنید "
            reshaped_txt = arabic_reshaper.reshape(msg)
            msg = get_display(reshaped_txt)
            message1 = messagebox.showinfo("هشدار", msg)
            if message1 == 'ok':
                E1.insert(nameVar.get())
                E2.insert(lastnameVar.get())
                E3.insert(fathernameVar.get())
                E4.insert(IDVar.get())

        elif val_gender not in (1, 2):
            msg = " جنسیت را مشخص کنید "
            reshaped_txt = arabic_reshaper.reshape(msg)
            msg = get_display(reshaped_txt)
            message1 = messagebox.showinfo("هشدار", msg)
            if message1 == 'ok':
                E1.insert(nameVar.get())
                E2.insert(lastnameVar.get())
                E3.insert(fathernameVar.get())
                E4.insert(IDVar.get())

        elif val_national not in (1, 2):
            msg = " ملیت را مشخص کنید "
            reshaped_txt = arabic_reshaper.reshape(msg)
            msg = get_display(reshaped_txt)
            message1 = messagebox.showinfo("هشدار", msg)
            if message1 == 'ok':
                E1.insert(nameVar.get())
                E2.insert(lastnameVar.get())
                E3.insert(fathernameVar.get())
                E4.insert(IDVar.get())


        elif len(text_ID) > 10 or len(text_ID) < 10:
            msg = "مقدار کد ملی باید ۱۰ کاراکتر داشته باشد"
            reshaped_txt = arabic_reshaper.reshape(msg)
            msg = get_display(reshaped_txt)
            message2 = messagebox.showinfo('هشدار', msg)
            if message2 == 'ok':
                E1.insert(nameVar.get())
                E2.insert(lastnameVar.get())
                E3.insert(fathernameVar.get())
                E4.insert(IDVar.get())


        elif len(text_path) == 0:
            msg = "عکس را بارگذاری کنید"
            reshaped_txt = arabic_reshaper.reshape(msg)
            msg = get_display(reshaped_txt)
            message3 = messagebox.showinfo('هشدار', msg)
            if message3 == 'ok':
                E1.insert(nameVar.get())
                E2.insert(lastnameVar.get())
                E3.insert(fathernameVar.get())
                E4.insert(IDVar.get())



        else:
            warning = "اطلاعات وارد شده صحیح است؟"
            reshaped_txt = arabic_reshaper.reshape(warning)
            warning = get_display(reshaped_txt)
            DB_Save = messagebox.askquestion("هشدار", warning)
            if DB_Save == 'yes':
                addnew()
                msg = "اطلاعات با موفقیت ذخیره شد"
                reshaped_txt = arabic_reshaper.reshape(msg)
                msg = get_display(reshaped_txt)
                messagebox.showinfo('message', msg)
                E1.delete(0, tkinter.END)
                E2.delete(0, tkinter.END)
                E3.delete(0, tkinter.END)
                E4.delete(0, tkinter.END)
                ent1.delete(0, tkinter.END)
                r1.deselect()
                r2.deselect()
                r3.deselect()
                r4.deselect()
                canvas_child.after(10, canvas_child.destroy)
                child.destroy()
                add()

    # -----------------------------------------------------------------
    def exit1():
        E1.delete(0, tkinter.END)
        E2.delete(0, tkinter.END)
        E3.delete(0, tkinter.END)
        E4.delete(0, tkinter.END)
        ent1.delete(0, tkinter.END)
        r1.deselect()
        r2.deselect()
        r3.deselect()
        r4.deselect()
        child.destroy()

    # buttons---------------------------------------------------------------------------------
    quitt = "خروج"
    reshaped_txt = arabic_reshaper.reshape(quitt)
    quitt = get_display(reshaped_txt)
    quitWin = tkinter.Button(child, text=quitt, command=exit1, fg="white", bg="#466a8d", width=3, height=1,
                             activebackground="white", font=('times', 16, ' bold '))
    quitWin.place(x=350, y=450, relwidth=0.15)

    savee = "ذخیره"
    reshaped_txt = arabic_reshaper.reshape(savee)
    savee = get_display(reshaped_txt)
    camerawin = tkinter.Button(child, text=savee, command=save, fg="white", bg="#466a8d", width=3, height=1,
                               activebackground="white", font=('times', 16, ' bold '))
    camerawin.place(x=780, y=450, relwidth=0.15)

    def browsefunc():
        filename = tkinter.filedialog.askopenfilename(filetypes=(
        ("image files", "*.jpg"), ('image files', '*.png'), ('image files', '*.jpeg'), ("All files", "*.*")))

        if len(ent1.get()) == 0:
            ent1.insert(END, filename)

        else:
            ent1.delete(0, tkinter.END)
            ent1.insert(END, filename)

        img = Image.open(filename)
        h , w  = img.size
        if (h,w)>=(250,250):
            # img = img.resize((250, 250), Image.ANTIALIAS)
            img = img.resize((250, 250), Image.BILINEAR)
            img = ImageTk.PhotoImage(img)

            panel = Label(canvas_child, image=img)
            panel.image = img
            panel.grid(row=2)
        else:
            msg = "ابعاد تصویر قابل قبول نمی باشد."
            reshaped_txt = arabic_reshaper.reshape(msg)
            msg = get_display(reshaped_txt)
            message4 = messagebox.showinfo('هشدار', msg)
            if message4 == 'ok':
                ent1.delete(0, tkinter.END)

    # -----------------------------------------------------

    img = "بارگذاری تصویر"
    reshaped_txt = arabic_reshaper.reshape(img)
    img = get_display(reshaped_txt)
    btn = Button(child, text=img, command=browsefunc, font=('times', 14, 'bold'))
    btn.place(x=40, y=410, relwidth=0.15)

    ent1 = Entry(child, textvar=imagepath, bd=5)
    ent1.place(x=330, y=340, relwidth=0.58, relheight=0.08)

    path = "مسیر عکس"
    reshaped_txt = arabic_reshaper.reshape(path)
    path = get_display(reshaped_txt)

    label1 = Label(child, text=path, font=('times', 16, 'bold'))
    label1.pack(padx=1, pady=20)
    label1.place(x=1000, y=340, relwidth=0.1)

    date = JalaliDate.today()
    label = Label(child, text=f"{date:%A, %B %d, %Y}", font="Calibri, 15")
    label.pack(padx=1, pady=20)
    label.place(x=380, y=40)


if __name__ == '__main__':
    conn = sqlite3.connect('/home/bagheri/PycharmProjects/pythonProject4/testUI2.db')
    cursor = conn.cursor()
    # first window------------------------------------------------
    parent = Tk()
    parent.title("سیستم تشخیص چهره")
    parent.geometry("1520x850")
    parent.resizable(0, 0)
    parent.configure(background='#7eb2f9')

    # cap = cv2.VideoCapture(0)
    cap = cv2.imread("/home/bagheri/Desktop/1300_1390.jpg")
    # cap =  cv2.VideoCapture("//home/jalali/Desktop/vlc-record-2023-01-17-09h33m41s-rtsp___192.168.1.57_user=root&password=1qazxsw2&channel=1&stream=0.sdp-.avi")
    # cap = cv2.VideoCapture(
    #     "rtsp://192.168.1.57/user=root&password=1qazxsw2&channel=1&stream=0.sdp?real_stream--rtp-caching=100")

    # Create torchvision model
    model_path = '/home/bagheri/PycharmProjects/pythonProject4/model.pt'
    return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    RetinaFace = create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(model_path)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}

    RetinaFace.load_state_dict(pretrained_dict)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    RetinaFace.to(device)
    RetinaFace.eval()

    cursor.execute(
        'CREATE TABLE IF NOT EXISTS personInfo_1 (Name CHAR(20) NOT NULL,family CHAR(30) NOT NULL,fname CHAR(20) NOT NULL,ID CHAR(10) PRIMARY KEY NOT NULL,gender CHAR(1) NOT NULL,national CHAR(1) NOT NULL, photo BLOB NOT NULL ,embedding CHAR(2622) NOT NULL)')

    face_model = VGGFace.loadModel()
    known_face_encodings, known_face_names = known_face()

    # frames-------------------------------------------------
    frame1 = tkinter.Frame(parent, bg="white")
    frame1.place(relx=0.08, rely=0.06, relwidth=0.55, relheight=0.85)

    frame2 = tkinter.Frame(parent, bg="white")
    frame2.place(relx=0.64, rely=0.06, relwidth=0.35, relheight=0.85)

    frame3 = tkinter.Frame(parent, bg="white")
    frame3.place(relx=0.64, rely=0.55, relwidth=0.35, relheight=0.3)

    frame4 = tkinter.Frame(parent, bg="#F0F0FF")
    frame4.place(relx=0, rely=0, relwidth=0.075, relheight=1)

    # frame_headder-----------------------------------------------------------------

    camera = "دوربین"
    reshaped_txt = arabic_reshaper.reshape(camera)
    camera = get_display(reshaped_txt)
    fr_head1 = tkinter.Label(frame1, text=camera, fg="white", bg="#0a4089", font=('times', 17, ' bold '))
    fr_head1.place(x=0, y=0, relwidth=1)

    detect = "آشکارسازی"
    reshaped_txt = arabic_reshaper.reshape(detect)
    detect = get_display(reshaped_txt)
    fr_head2 = tkinter.Label(frame2, text=detect, fg="white", bg="#0a4089", font=('times', 17, ' bold '))
    fr_head2.place(x=0, y=0, relwidth=1)

    recognize = "شناسایی"
    reshaped_txt = arabic_reshaper.reshape(recognize)
    recognize = get_display(reshaped_txt)
    fr_head3 = tkinter.Label(frame3, text=recognize, fg="white", bg="#0a4089", font=('times', 17, ' bold '))
    fr_head3.place(x=0, y=0, relwidth=1)

    menu = "نوار ابزار"
    reshaped_txt = arabic_reshaper.reshape(menu)
    menu = get_display(reshaped_txt)
    fr_head4 = tkinter.Label(frame4, text=menu, fg="white", bg="#0a4089", font=('times', 17, ' bold '))
    fr_head4.place(x=0, y=0, relwidth=1)

    date = JalaliDate.today()
    label = Label(parent, text=f"{date:%A, %B %d, %Y}", font="Calibri, 15")
    label.pack(padx=1, pady=10)

    lbl = Label(parent, font=('calibri', 15, 'bold'),
                background='#5454FF',
                foreground='white')

    lbl.pack()
    lbl.place(x=1200, y=10, relwidth=0.2)
    time()

    nameVar = StringVar()
    lastnameVar = StringVar()
    fathernameVar = StringVar()
    imagepath = StringVar()
    IDVar = StringVar()
    genderVar = IntVar()
    nationalVar = IntVar()

    # detection boxes---------------------------------------------------
    canvas1 = Canvas(parent, width=100, height=100)
    canvas1.pack()
    canvas1.place(x=1000, y=90, relwidth=0.07)
    a = canvas1.create_rectangle(0, 0, 0, 0, fill='black')
    canvas1.move(a, 200, 200)

    canvas2 = Canvas(parent, width=100, height=100)
    canvas2.pack()
    canvas2.place(x=1120, y=90, relwidth=0.07)
    a2 = canvas2.create_rectangle(0, 0, 0, 0, fill='black')
    canvas2.move(a2, 200, 200)

    canvas3 = Canvas(parent, width=100, height=100)
    canvas3.pack()
    canvas3.place(x=1240, y=90, relwidth=0.07)
    a3 = canvas3.create_rectangle(0, 0, 0, 0, fill='black')
    canvas3.move(a3, 200, 200)

    canvas4 = Canvas(parent, width=100, height=100)
    canvas4.pack()
    canvas4.place(x=1360, y=90, relwidth=0.07)
    a4 = canvas4.create_rectangle(0, 0, 0, 0, fill='black')
    canvas4.move(a4, 200, 200)

    canvas5 = Canvas(parent, width=100, height=100)
    canvas5.pack()
    canvas5.place(x=1000, y=210, relwidth=0.07)
    a5 = canvas5.create_rectangle(0, 0, 0, 0, fill='black')
    canvas5.move(a5, 200, 200)

    canvas6 = Canvas(parent, width=100, height=100)
    canvas6.pack()
    canvas6.place(x=1120, y=210, relwidth=0.07)
    a6 = canvas6.create_rectangle(0, 0, 0, 0, fill='black')
    canvas6.move(a6, 200, 200)

    canvas7 = Canvas(parent, width=100, height=100)
    canvas7.pack()
    canvas7.place(x=1240, y=210, relwidth=0.07)
    a7 = canvas7.create_rectangle(0, 0, 0, 0, fill='black')
    canvas7.move(a7, 200, 200)

    canvas8 = Canvas(parent, width=100, height=100)
    canvas8.pack()
    canvas8.place(x=1360, y=210, relwidth=0.07)
    a8 = canvas8.create_rectangle(0, 0, 0, 0, fill='black')
    canvas8.move(a8, 200, 200)

    canvas9 = Canvas(parent, width=100, height=100)
    canvas9.pack()
    canvas9.place(x=1000, y=330, relwidth=0.07)
    a9 = canvas9.create_rectangle(0, 0, 0, 0, fill='black')
    canvas9.move(a9, 200, 200)

    canvas10 = Canvas(parent, width=100, height=100)
    canvas10.pack()
    canvas10.place(x=1120, y=330, relwidth=0.07)
    a10 = canvas10.create_rectangle(0, 0, 0, 0, fill='black')
    canvas10.move(a10, 200, 200)

    canvas11 = Canvas(parent, width=100, height=100)
    canvas11.pack()
    canvas11.place(x=1240, y=330, relwidth=0.07)
    a11 = canvas11.create_rectangle(0, 0, 0, 0, fill='black')
    canvas11.move(a11, 200, 200)

    canvas12 = Canvas(parent, width=100, height=100)
    canvas12.pack()
    canvas12.place(x=1360, y=330, relwidth=0.07)
    a12 = canvas12.create_rectangle(0, 0, 0, 0, fill='black')
    canvas12.move(a12, 200, 200)
    canvases = [canvas1, canvas2, canvas3, canvas4, canvas5, canvas6, canvas7, canvas8, canvas9, canvas10, canvas11,
                canvas12]

    # recognition boxes---------------------------------------------------

    canvas13 = Canvas(parent, width=100, height=120)
    canvas13.pack()
    canvas13.place(x=980, y=500, relwidth=0.1, relheight=0.2)
    a13 = canvas13.create_rectangle(0, 0, 0, 0, fill='black')
    canvas13.move(a13, 200, 200)

    canvas14 = Canvas(parent, width=100, height=120)
    canvas14.pack()
    canvas14.place(x=1160, y=500, relwidth=0.1, relheight=0.2)
    a14 = canvas14.create_rectangle(0, 0, 0, 0, fill='black')
    canvas14.move(a14, 200, 200)

    canvas15 = Canvas(parent, width=100, height=120)
    canvas15.pack()
    canvas15.place(x=1340, y=500, relwidth=0.1, relheight=0.2)
    a15 = canvas15.create_rectangle(0, 0, 0, 0, fill='black')
    canvas15.move(a15, 200, 200)

    canvas16 = Canvas(parent, width=100, height=120)
    canvas16.pack()
    canvas16.place(x=980, y=680, relwidth=0.1, relheight=0.05)
    a16 = canvas16.create_rectangle(0, 0, 0, 0, fill='black')
    canvas16.move(a16, 200, 200)

    canvas17 = Canvas(parent, width=100, height=120)
    canvas17.pack()
    canvas17.place(x=1160, y=680, relwidth=0.1, relheight=0.05)
    a17 = canvas17.create_rectangle(0, 0, 0, 0, fill='black')
    canvas17.move(a17, 200, 200)

    canvas18 = Canvas(parent, width=100, height=120)
    canvas18.pack()
    canvas18.place(x=1340, y=680, relwidth=0.1, relheight=0.05)
    a18 = canvas18.create_rectangle(0, 0, 0, 0, fill='black')
    canvas18.move(a18, 200, 200)

    # button------------------------------------------------------------

    camm = "دوربین"
    reshaped_txt = arabic_reshaper.reshape(camm)
    camm = get_display(reshaped_txt)
    cammWin = tkinter.Button(frame4, text=camm, command=show_camera, fg="white", bg="#466a8d", width=3, height=1,
                             activebackground="white", font=('times', 16, ' bold '))
    cammWin.place(x=5, y=50, relwidth=0.8)

    detect = "آشکارسازی"
    reshaped_txt = arabic_reshaper.reshape(detect)
    detect = get_display(reshaped_txt)
    detectWin = tkinter.Button(frame4, text=detect, command=show_frames, fg="white", bg="#466a8d", width=3, height=1,
                               activebackground="white", font=('times', 16, ' bold '))
    detectWin.place(x=5, y=100, relwidth=0.8)

    rec = "شناسایی"
    reshaped_txt = arabic_reshaper.reshape(rec)
    rec = get_display(reshaped_txt)
    recWin = tkinter.Button(frame4, text=rec, command=recognition, fg="white", bg="#466a8d", width=3, height=1,
                            activebackground="white", font=('times', 16, ' bold '))
    recWin.place(x=5, y=150, relwidth=0.8)

    adding = "افزودن"
    reshaped_txt = arabic_reshaper.reshape(adding)
    adding = get_display(reshaped_txt)
    buttonWin = tkinter.Button(frame4, text=adding, command=add, fg="white", bg="#466a8d", width=3, height=1,
                               activebackground="white", font=('times', 16, ' bold '))
    buttonWin.place(x=5, y=200, relwidth=0.8)

    contct = "ارتباط با ما"
    reshaped_txt = arabic_reshaper.reshape(contct)
    contct = get_display(reshaped_txt)
    contactWin = tkinter.Button(frame4, text=contct, command=contact, fg="white", bg="#466a8d", width=3, height=1,
                                activebackground="white", font=('times', 16, ' bold '))
    contactWin.place(x=5, y=730, relwidth=0.8)

    exitt = "خروج"
    reshaped_txt = arabic_reshaper.reshape(exitt)
    exitt = get_display(reshaped_txt)
    quitWin = tkinter.Button(frame4, text=exitt, command=on_closing, fg="white", bg="#466a8d", width=3, height=1,
                             activebackground="white", font=('times', 16, ' bold '))
    quitWin.place(x=5, y=780, relwidth=0.8)

    # closing lines------------------------------------------------
    parent.protocol("WM_DELETE_WINDOW", on_closing)
    parent.mainloop()