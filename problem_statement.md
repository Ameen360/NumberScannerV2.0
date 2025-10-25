# AI-Powered Automated Car Plate Number Recognition System

## Problem Statement

This project aims to develop a mobile application that can:
1. Extract alphanumeric text from car license plates using OCR (already working)
2. Identify the type of plate based on its background color:
   - Green: Government vehicles
   - Blue: Private vehicles
   - Red: Commercial vehicles

## Research Gap

Current license plate recognition systems typically focus only on extracting the alphanumeric characters from license plates. This project extends that functionality by adding color-based classification to determine the vehicle category, which provides additional contextual information that can be valuable for:

- Law enforcement agencies to quickly identify vehicle types
- Automated tolling systems to apply different rates based on vehicle categories
- Parking management systems to allocate spaces based on vehicle types
- Traffic monitoring and analysis

## Technical Approach

The system will be implemented using a two-stage pipeline:
1. OCR Module: Extract the alphanumeric plate number (already implemented)
2. Color Classification Module: Identify plate category by analyzing the dominant background color

This approach allows for modular development and testing, with the final system integrating both components for a complete solution.