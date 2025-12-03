from .teacher_model import TeacherModel
from .student_model import StudentModel
from config.config import MODEL_MAPPING

def get_model(model_type, num_classes):
    """根据配置自动返回模型实例（无需修改，脚本调用此函数加载模型）"""
    model_class_name = MODEL_MAPPING.get(model_type)
    if model_class_name == "TeacherModel":
        return TeacherModel(num_classes=num_classes)
    elif model_class_name == "StudentModel":
        return StudentModel(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}，请在MODEL_MAPPING中添加")