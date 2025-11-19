import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { router as aiRoutes } from './routes/aiRoutes.js'; // FIXED: Imports the named export 'router' and renames it to 'aiRoutes'
import cors from 'cors'; // Added CORS import

dotenv.config(); // Load environment variables from .env

const app = express();
const PORT = process.env.PORT || 3000;
const MONGODB_URI = process.env.MONGO_URI; // Correctly reads MONGO_URI

// Middleware
app.use(express.json()); // To parse JSON request bodies
app.use(cors()); // Enable CORS

// MongoDB Connection
mongoose.connect(MONGODB_URI)
    .then(() => console.log('✅ MongoDB connected successfully.'))
    .catch(err => console.error('❌ MongoDB connection error:', err));

// Routes
app.use('/api/ai', aiRoutes); // Use routes with a base path

// Start Server
app.listen(PORT, () => {
    console.log(`[Express] 🟢 Server running on http://localhost:${PORT}`);
    // Show the target URL for debugging
    console.log(`[Express] Targeting Flask at: ${process.env.AI_API_URL}`); 
});
