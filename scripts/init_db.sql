-- MindRouter2 Database Initialization
-- This script runs on first container startup

-- Ensure the database exists with proper character set
CREATE DATABASE IF NOT EXISTS mindrouter
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Grant privileges to the application user
GRANT ALL PRIVILEGES ON mindrouter.* TO 'mindrouter'@'%';
FLUSH PRIVILEGES;

-- Note: Tables are created by Alembic migrations, not here
-- This script only ensures the database exists with proper encoding
