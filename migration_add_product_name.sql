-- 向model_trainings表添加product_name列
ALTER TABLE model_trainings ADD COLUMN product_name VARCHAR(50);
-- 创建索引以提高查询性能
CREATE INDEX idx_model_trainings_product_name ON model_trainings (product_name);

-- 向model_evaluations表添加product_name列
ALTER TABLE model_evaluations ADD COLUMN product_name VARCHAR(50);
-- 创建索引以提高查询性能
CREATE INDEX idx_model_evaluations_product_name ON model_evaluations (product_name);

-- 更新现有记录，从关联的蔬菜数据中填充product_name (针对model_trainings)
UPDATE model_trainings 
SET product_name = (
    SELECT product_name 
    FROM vegetables 
    WHERE vegetables.id = model_trainings.vegetable_id
)
WHERE vegetable_id IS NOT NULL;

-- 更新现有记录，从关联的蔬菜数据中填充product_name (针对model_evaluations)
UPDATE model_evaluations 
SET product_name = (
    SELECT product_name 
    FROM vegetables 
    WHERE vegetables.id = model_evaluations.vegetable_id
)
WHERE vegetable_id IS NOT NULL; 