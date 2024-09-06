#!/bin/bash


GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN} Starting LCM type generation...${NC}"

# Make
lcm-gen -p *.lcm .

echo -e "${GREEN} Done with LCM type generation${NC}"
