package com.example.mcpserver.config;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Manages server configuration
 */
public class ConfigManager {
    private static final Logger LOGGER = LogManager.getLogger(ConfigManager.class);
    
    private Properties properties;
    private File configFile;
    
    public ConfigManager(String configPath) {
        this.configFile = new File(configPath);
        this.properties = new Properties();
    }
    
    /**
     * Loads configuration from file
     */
    public void loadConfig() {
        LOGGER.info("Loading configuration from " + configFile.getAbsolutePath());
        
        try (FileInputStream fis = new FileInputStream(configFile)) {
            properties.load(fis);
            LOGGER.info("Configuration loaded successfully with " + properties.size() + " properties");
        } catch (IOException e) {
            LOGGER.error("Failed to load configuration", e);
            // Generate default configuration
            generateDefaultConfig();
        }
    }
    
    /**
     * Generates default configuration values
     */
    private void generateDefaultConfig() {
        LOGGER.info("Generating default configuration");
        
        // Set default values
        properties.setProperty("server-port", "25565");
        properties.setProperty("max-players", "20");
        properties.setProperty("view-distance", "10");
        properties.setProperty("motd", "MCP Development Server");
        properties.setProperty("level-name", "world");
        
        // Additional default properties would be set here
        
        LOGGER.info("Default configuration generated");
    }
    
    /**
     * Gets a string property
     */
    public String getProperty(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    /**
     * Gets an integer property
     */
    public int getIntProperty(String key, int defaultValue) {
        String value = properties.getProperty(key);
        
        if (value == null) {
            return defaultValue;
        }
        
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            LOGGER.warn("Invalid integer property value for key '" + key + "': " + value);
            return defaultValue;
        }
    }
    
    /**
     * Gets a boolean property
     */
    public boolean getBooleanProperty(String key, boolean defaultValue) {
        String value = properties.getProperty(key);
        
        if (value == null) {
            return defaultValue;
        }
        
        return Boolean.parseBoolean(value);
    }
}
