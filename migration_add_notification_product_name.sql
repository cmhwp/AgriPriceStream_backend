-- 为notifications表添加product_name字段
ALTER TABLE notifications ADD COLUMN product_name VARCHAR(50) NULL;

-- 更新现有数据：将vegetable_id对应的product_name填充到新字段中
UPDATE notifications n
SET product_name = (
    SELECT v.product_name
    FROM vegetables v
    WHERE v.id = n.vegetable_id
);

-- 添加索引提高查询性能
CREATE INDEX idx_notifications_product_name ON notifications(product_name);

-- 注意：保留vegetable_id字段以维持向后兼容性
-- 未来版本可以考虑设置product_name为NOT NULL并移除vegetable_id 