#!/bin/bash
# ShopBis Dashboard Startup Script for Render
# This script ensures the app runs from the correct directory

cd Dashboard
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
