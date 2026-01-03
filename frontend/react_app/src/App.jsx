// frontend/react_app/src/App.jsx
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/common/Layout';
import GenerationPanel from './components/generation/GenerationPanel';
import LoRAManager from './components/lora/LoRAManager';
import BatchProcessor from './components/batch/BatchProcessor';
import TrainingMonitor from './components/training/TrainingMonitor';
import ImageGallery from './components/gallery/ImageGallery';
import T2IJobs from './components/jobs/T2IJobs';
import apiService from './services/apiService';
import './styles/components/App.css';

function App() {
  useEffect(() => {
    apiService.bootstrapJwtSession({ silent: true }).catch(() => {});
  }, []);

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
            <Route path="/jobs" element={<T2IJobs />} />
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
