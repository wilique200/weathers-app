# Use official Node image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy frontend files
COPY . /app

# Install dependencies
RUN npm install

# Build production files
RUN npm run build

# Expose React port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]