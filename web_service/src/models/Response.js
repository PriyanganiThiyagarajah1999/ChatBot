import mongoose from 'mongoose';

const ResponseSchema = new mongoose.Schema({
    prompt: {
        type: String,
        required: true
    },
    aiResponse: {
        type: String,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

export default mongoose.model('Response', ResponseSchema);