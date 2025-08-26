// frontend/react_app/src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/common/Layout';
import GenerationPanel from './components/generation/GenerationPanel';
import LoRAManager from './components/lora/LoRAManager';
import BatchProcessor from './components/batch/BatchProcessor';
import TrainingMonitor from './components/training/TrainingMonitor';
import ImageGallery from './components/gallery/ImageGallery';
import './styles/components/App.css';

function App() {
  return (
    <div className="App">
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<GenerationPanel />} />
            <Route path="/generation" element={<GenerationPanel />} />
            <Route path="/lora" element={<LoRAManager />} />
            <Route path="/batch" element={<BatchProcessor />} />
            <Route path="/training" element={<TrainingMonitor />} />
            <Route path="/gallery" element={<ImageGallery />} />
          </Routes>
        </Layout>
      </Router>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
    </div>
  );
}

export default App;