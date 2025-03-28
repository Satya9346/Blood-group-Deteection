import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Droplet, Upload, HomeIcon, Info, Settings } from 'lucide-react';
import Home from './pages/Home';
import About from './pages/About';
import Services from './pages/Services';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
        <nav className="bg-white shadow-md">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <Droplet className="h-8 w-8 text-red-500" />
                <span className="ml-2 text-xl font-bold text-gray-800">BloodPrint</span>
              </div>
              <div className="flex items-center space-x-8">
                <Link to="/" className="text-gray-600 hover:text-gray-900 flex items-center">
                  <HomeIcon className="h-4 w-4 mr-1" />
                  Home
                </Link>
                <Link to="/services" className="text-gray-600 hover:text-gray-900 flex items-center">
                  <Settings className="h-4 w-4 mr-1" />
                  Services
                </Link>
                <Link to="/about" className="text-gray-600 hover:text-gray-900 flex items-center">
                  <Info className="h-4 w-4 mr-1" />
                  About
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/services" element={<Services />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;