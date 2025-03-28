import React from 'react';
import { Users, Target, Heart } from 'lucide-react';

function About() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-16">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">About BloodPrint</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Revolutionizing blood group detection through cutting-edge fingerprint analysis technology
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-12 items-center mb-16">
        <div>
          <img
            src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=2400&q=80"
            alt="Medical Technology"
            className="rounded-2xl shadow-lg"
          />
        </div>
        <div>
          <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
          <p className="text-gray-600 mb-6">
            At BloodPrint, we're dedicated to transforming healthcare through innovative technology.
            Our groundbreaking approach to blood group detection combines advanced pattern recognition
            with machine learning to provide accurate, non-invasive blood typing through fingerprint analysis.
          </p>
          <p className="text-gray-600">
            This technology represents a significant breakthrough in medical diagnostics, offering a
            faster, more comfortable alternative to traditional blood testing methods while maintaining
            the highest standards of accuracy and reliability.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
        <div className="text-center p-6">
          <Users className="h-12 w-12 text-blue-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Expert Team</h3>
          <p className="text-gray-600">
            Led by renowned researchers and medical professionals with decades of experience
          </p>
        </div>
        <div className="text-center p-6">
          <Target className="h-12 w-12 text-green-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Precision Focus</h3>
          <p className="text-gray-600">
            Committed to delivering accurate results through rigorous testing and validation
          </p>
        </div>
        <div className="text-center p-6">
          <Heart className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Patient Care</h3>
          <p className="text-gray-600">
            Prioritizing patient comfort and convenience in medical diagnostics
          </p>
        </div>
      </div>

      <div className="bg-gray-50 rounded-2xl p-8">
        <h2 className="text-2xl font-bold mb-6">Technology Overview</h2>
        <p className="text-gray-600 mb-4">
          Our proprietary algorithm analyzes the unique patterns in fingerprints that correlate with
          blood group markers. This revolutionary approach has been validated through extensive
          clinical trials and peer-reviewed research, demonstrating remarkable accuracy in blood
          group prediction.
        </p>
        <p className="text-gray-600">
          The system employs advanced image processing and machine learning techniques to identify
          subtle correlations between dermatoglyphic patterns and blood group antigens, providing
          results in minutes without the need for traditional blood sampling.
        </p>
      </div>
    </div>
  );
}

export default About;