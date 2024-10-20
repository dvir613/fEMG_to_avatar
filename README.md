# fEMG to Avatar
# Overview
This project aims to predict Facial Action Units (AUs) using facial electromyography (fEMG) recorded from surface electrodes and visualize the predictions on 3D avatars presented in Unity and creaated using ICTFace kit in Maya. The repository contains all necessary data processing scripts, Python-to-C# communication scripts, and Unity nece files, including the FBX files of the avatars.

# Introduction
Facial electromyography (fEMG) is a technique used to measure muscle activity by detecting electrical potentials generated by muscle cells when they contract. This project leverages fEMG signals to predict Facial Action Units (AUs) and visualize the movements on 3D avatars in real-time using Unity.

The motivation behind this project is to aid in the treatment of facial palsy by providing a better therapeutic experience and avoiding the discomfort patients feel when looking at their asymmetrical faces in the mirror.

# Background
Facial Action Coding System (FACS) is a comprehensive, anatomically-based system for describing all observable facial movements. Developed by Paul Ekman and Wallace V. Friesen in the 1970s, FACS defines action units (AUs) as the fundamental actions of individual muscles or groups of muscles. In animation and computer graphics, these AUs are often used to create realistic facial expressions by manipulating the blend shapes of 3D avatars. By accurately predicting AUs from fEMG signals, we can achieve natural and expressive animations that reflect true muscle activity.

# Features
* Real-time fEMG data acquisition
* Prediction of Facial Action Units
* Real-time visualization on 3D avatars in Unity
* Python-to-C# data communication

