import express from 'express';
import axios from 'axios';
import Response from '../models/Response.js'; // Imports the default export from the Response model

const router = express.Router();

// IMPORTANT: This reads the full URL including /chat from the .env file (AI_API_URL)
const FLASK_API_URL = process.env.AI_API_URL || 'http://127.0.0.1:5050/chat'; 

/**
 * POST /api/ai/generate
 * Takes a prompt, calls the Flask AI service, and saves the response to MongoDB.
 */
router.post('/generate', async (req, res) => {
    // Expected incoming client body: { "prompt": "..." }
    const { prompt } = req.body; 

    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required (key: "prompt").' });
    }

    try {
        // STEP 1: Call the Flask AI service
        console.log(`[Express] Calling Flask AI service at: ${FLASK_API_URL}`);
        
        const flaskResponse = await axios.post(FLASK_API_URL, {
            // Flask app.py expects the input key to be "message"
            message: prompt 
        });
        
        // Flask app.py returns the output key as "reply"
        const aiResponseText = flaskResponse.data.reply;

        // STEP 2: Save the interaction to MongoDB
        const newResponse = new Response({
            prompt: prompt,
            aiResponse: aiResponseText
        });
        const savedResponse = await newResponse.save();
        console.log(`[Express] Saved response ID: ${savedResponse._id}`);

        // STEP 3: Respond to the client
        res.status(200).json({
            message: 'AI response generated and saved successfully.',
            data: savedResponse
        });

    } catch (error) {
        console.error('Error communicating with Flask or MongoDB:', error.message);
        
        if (error.code === 'ECONNREFUSED' || !error.response) {
             console.error('[Express] AI Service Connection Error: Flask is unreachable.');
             return res.status(503).json({ 
                error: 'AI Service (Flask) is currently unavailable on 5050.', 
                details: error.message 
            });
        }
        console.error('[Express] Flask Error Response Data:', error.response?.data);
        res.status(500).json({ error: 'Internal server error', details: error.message });
    }
});

// CRITICAL: Export using the named export 'router' to match server.js import
export { router }; 
