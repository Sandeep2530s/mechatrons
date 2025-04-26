const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');
const mongoose = require('mongoose');
require('dotenv').config();

const app = express();
const PORT = 5001;

app.use(bodyParser.json());
app.use(cors());

console.log(`🚀 Server starting on http://localhost:${PORT}`);

// ✅ Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('✅ Connected to MongoDB'))
.catch(err => console.error('❌ MongoDB Connection Error:', err));

// ✅ MongoDB Schema & Models
const urlSchema = new mongoose.Schema({
    url: String,
    prediction: String,
    timestamp: { type: Date, default: Date.now },
});
const URLRecord = mongoose.model('URLRecord', urlSchema);

const smsSchema = new mongoose.Schema({
    sms: String,
    prediction: String,
    timestamp: { type: Date, default: Date.now },
});
const SMSRecord = mongoose.model('SMSRecord', smsSchema);

// ✅ Serve frontend
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ✅ POST: Check Phishing URL and Store
app.post('/check-url', async (req, res) => {
    const { url } = req.body;
    console.log("✅ Checking URL:", url);

    if (!url) return res.status(400).json({ error: 'URL is required' });

    const options = {
        mode: 'text',
        pythonPath: 'python',
        pythonOptions: ['-u'],
        scriptPath: __dirname,
        args: [url],
    };

    let messages = [];

    const pyshell = new PythonShell("predict.py", options);

    pyshell.on('message', (message) => {
        console.log("📌 Python Output:", message);
        messages.push(message.trim());
    });

    pyshell.end(async (err) => {
        if (err || messages.length === 0) {
            console.error("❌ Error Processing Python Script:", err);
            return res.status(500).json({ error: 'Error processing request' });
        }

        const prediction = messages.pop();
        const responseMessage = prediction === "1" ? "Phishing" : "Safe";

        try {
            const newEntry = new URLRecord({ url, prediction: responseMessage });
            await newEntry.save();
            console.log("✅ Saved to MongoDB:", newEntry);
        } catch (error) {
            console.error("❌ MongoDB Save Error:", error);
            return res.status(500).json({ error: 'Database error' });
        }

        res.json({ url, prediction: responseMessage });
    });
});

// ✅ GET: Retrieve recent stored URLs
app.get('/stored-urls', async (req, res) => {
    try {
        const records = await URLRecord.find().sort({ timestamp: -1 }).limit(10);
        res.json(records);
    } catch (err) {
        res.status(500).json({ error: 'Error retrieving data' });
    }
});

// ✅ DELETE: Remove a URL record
app.delete('/delete-url/:id', async (req, res) => {
    try {
        await URLRecord.findByIdAndDelete(req.params.id);
        res.json({ message: 'Deleted successfully' });
    } catch (err) {
        res.status(500).json({ error: 'Error deleting record' });
    }
});

// ✅ POST: Check SMS and Store
app.post('/check-sms', async (req, res) => {
    const { sms } = req.body;
    console.log("✅ Checking SMS:", sms);

    if (!sms) return res.status(400).json({ error: 'SMS is required' });

    const options = {
        mode: 'text',
        pythonPath: 'python',
        pythonOptions: ['-u'],
        scriptPath: __dirname,
        args: [sms],
    };

    let messages = [];

    const pyshell = new PythonShell("predict_sms.py", options);

    pyshell.on('message', (message) => {
        console.log("📌 Python Output:", message);
        messages.push(message.trim());
    });

    pyshell.end(async (err) => {
        if (err || messages.length === 0) {
            console.error("❌ Error Processing Python Script:", err);
            return res.status(500).json({ error: 'Error processing request' });
        }

        const prediction = messages.pop();
        const responseMessage = prediction === "1" ? "Spam" : "Not Spam";

        try {
            const newEntry = new SMSRecord({ sms, prediction: responseMessage });
            await newEntry.save();
            console.log("✅ Saved to MongoDB:", newEntry);
        } catch (error) {
            console.error("❌ MongoDB Save Error:", error);
            return res.status(500).json({ error: 'Database error' });
        }

        res.json({ sms, prediction: responseMessage });
    });
});

// ✅ GET: Retrieve recent stored SMS
app.get('/stored-sms', async (req, res) => {
    try {
        const records = await SMSRecord.find().sort({ timestamp: -1 }).limit(10);
        res.json(records);
    } catch (err) {
        res.status(500).json({ error: 'Error retrieving data' });
    }
});

// ✅ DELETE: Remove an SMS record
app.delete('/delete-sms/:id', async (req, res) => {
    try {
        await SMSRecord.findByIdAndDelete(req.params.id);
        res.json({ message: 'Deleted successfully' });
    } catch (err) {
        res.status(500).json({ error: 'Error deleting record' });
    }
});

// ✅ Start Server
app.listen(PORT, () => {
    console.log(`✅ Server running at http://localhost:${PORT}`);
});
