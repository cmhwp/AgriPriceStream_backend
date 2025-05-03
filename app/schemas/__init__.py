# Make the db package importable 
from app.schemas.user import *
from app.schemas.vegetable import *
from app.schemas.price import *
from app.schemas.prediction import *
from app.schemas.dashboard import *
from app.schemas.response import *

# 注意：不再需要手动使用model_rebuild()
# 在Pydantic v2中，循环引用会自动处理

# 为向后兼容，修改相关引用
from app.schemas.vegetable import VegetableDetail
from app.schemas.price import PriceRecordResponse 