-- 15_entity_type.sql
-- Add 'Entity' to the artifact_type enum for NER-extracted entities.
-- The semantic entity type (Person, Organization, etc.) is stored in
-- kg_nodes.properties->>'entity_type'.

ALTER TYPE artifact_type ADD VALUE IF NOT EXISTS 'Entity';
