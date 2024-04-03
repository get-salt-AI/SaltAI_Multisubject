import os
import copy
import nodes
import importlib
import numpy as np
import insightface
import folder_paths
from . import utils

try:
    import torch.cuda as cuda
except:
    cuda = None
if cuda is not None:
    if cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
reactor = importlib.import_module('comfyui-reactor-node')
models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
ANALYSIS_MODEL = None


def getAnalysisModel():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    return ANALYSIS_MODEL


def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = copy.deepcopy(getAnalysisModel())
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser.get(img_data)


def try_install_custom_node(custom_node_url, msg):
    try:
        import cm_global
        cm_global.try_call(
            api='cm.try-install-custom-node',
            sender="Inspire Pack",
            custom_node_url=custom_node_url,
            msg=msg,
        )
    except Exception:
        print(msg)
        print(f"[Inspire Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")


class MultisubjectFaceSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "face_areas": ("FACE_AREAS", {}),
                "swap_model": (list(reactor.nodes.model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (reactor.nodes.get_model_names(reactor.nodes.get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1, "step": 0.05}),
                "face0": ("IMAGE", {}),
                "face1": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "multisubject_face_swap"
    CATEGORY = "SALT/FaceSwap"

    def multisubject_face_swap(
        self,
        image,
        face_areas,
        swap_model,
        facedetection,
        face_restore_model,
        face_restore_visibility,
        codeformer_weight,
        **kwargs
    ):
        if 'ReActorFaceSwap' not in nodes.NODE_CLASS_MAPPINGS:
            try_install_custom_node('https://github.com/Gourieff/comfyui-reactor-node',
                                    "To use 'MultisubjectFaceSwap' node, 'ReActor' extension is required.")
            raise Exception(f"[ERROR] To use MultisubjectFaceSwap, you need to install ReActor node pack.")
        face_swapper = nodes.NODE_CLASS_MAPPINGS['ReActorFaceSwap']()
        
        faces = kwargs.values()
        enabled = True
        detect_gender_input = "no"
        detect_gender_source = "no"

        founded_faces = analyze_faces(np.array(utils.batch_tensor_to_pil(image)[0], dtype=np.uint8))
        founded_faces = sorted(founded_faces, key=lambda x: x.bbox[0])
        founded_faces = [f.bbox for f in founded_faces]
        face_areas = [(f[1] * image.shape[2], f[0]* image.shape[1], f[3]* image.shape[2], f[2]* image.shape[1]) for f in face_areas]
        
        ious = []
        for idx in range(len(founded_faces)):
            ious.append([utils.calculate_iou(fa, founded_faces[idx]) for fa in face_areas])
        best_matches = np.argmax(ious, axis=0)
        
        for face, index in zip(faces, best_matches[:len(faces)]):
            input_faces_index = str(index)
            source_faces_index = "0"
            console_log_level = 1
            image, _ = face_swapper.execute(
                enabled,
                image, 
                swap_model,
                detect_gender_source,
                detect_gender_input,
                source_faces_index,
                input_faces_index,
                console_log_level,
                face_restore_model,
                face_restore_visibility,
                codeformer_weight,
                facedetection,
                source_image=face,
            )
        return (image,)
