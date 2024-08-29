
# relevant blendshapes from LiveCapture without any of the left side of the face
relevant_blendshapes = ['BrowDownRight', 'BrowInnerUp', 'BrowOuterUpRight', 'CheekPuff',
                        'CheekSquintRight', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight',
                        'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawOpen', 'JawRight', 'MouthClose',
                        'MouthDimpleRight', 'MouthFrownRight', 'MouthFunnel', 'MouthLowerDownRight', 'MouthPressRight',
                        'MouthPucker', 'MouthRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
                        'MouthSmileRight', 'MouthStretchRight', 'MouthUpperUpRight', 'NoseSneerRight']

# mapping from LiveCapture to the custom-made avatar
mapping = {
    "BrowInnerUp": "BrowInnerUpRight",
    "CheekPuff": "CheekPuffRight",
}

# Define the desired order of the columns that will match the order of the blendshapes in the custom-made avatar
blend_shapes = [
    "BrowDownLeft", "BrowDownRight",
    "BrowInnerUpLeft", "BrowInnerUpRight",
    "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuffLeft", "CheekPuffRight",
    "CheekRaiserLeft", "CheekRaiserRight",
    "CheekSquintLeft", "CheekSquintRight",
    "EyeBlinkLeft", "EyeBlinkRight",
    "EyeLookDownLeft", "EyeLookDownRight",
    "EyeLookInLeft", "EyeLookInRight",
    "EyeLookOutLeft", "EyeLookOutRight",
    "EyeLookUpLeft", "EyeLookUpRight",
    "EyeSquintLeft", "EyeSquintRight",
    "EyeWideLeft", "EyeWideRight",
    "JawForward",
    "JawLeft",
    "JawOpen",
    "JawRight",
    "MouthClose",
    "MouthDimpleLeft", "MouthDimpleRight",
    "MouthFrownLeft", "MouthFrownRight",
    "MouthFunnel",
    "MouthLeft",
    "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthPressLeft", "MouthPressRight",
    "MouthPucker",
    "MouthRight",
    "MouthRollLower", "MouthRollUpper",
    "MouthShrugLower", "MouthShrugUpper",
    "MouthSmileLeft", "MouthSmileRight",
    "MouthStretchLeft", "MouthStretchRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "NoseSneerLeft", "NoseSneerRight",
    "PupilDilateLeft", "PupilDilateRight"
]




