import React from 'react';
import { Shield, Clock, CheckCircle, Award } from 'lucide-react';

function Services() {
  const services = [
    {
      icon: <Shield className="h-8 w-8 text-blue-500" />,
      title: "Advanced Pattern Recognition",
      description: "State-of-the-art AI algorithms analyze fingerprint patterns to determine blood group with high accuracy."
    },
    {
      icon: <Clock className="h-8 w-8 text-green-500" />,
      title: "Rapid Results",
      description: "Get blood group results in minutes, eliminating the need for traditional blood testing methods."
    },
    {
      icon: <CheckCircle className="h-8 w-8 text-purple-500" />,
      title: "Non-Invasive Testing",
      description: "Complete blood group analysis without the need for blood samples, ensuring comfort and convenience."
    },
    {
      icon: <Award className="h-8 w-8 text-yellow-500" />,
      title: "Certified Accuracy",
      description: "Our system has been validated through extensive clinical trials with 90.0% accuracy rate."
    }
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold text-center text-gray-900 mb-12">Our Services</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {services.map((service, index) => (
          <div key={index} className="bg-white rounded-xl shadow-lg p-8 transform hover:scale-105 transition-transform">
            <div className="flex items-center mb-4">
              {service.icon}
              <h3 className="text-xl font-semibold ml-3">{service.title}</h3>
            </div>
            <p className="text-gray-600">{service.description}</p>
          </div>
        ))}
      </div>

      <div className="mt-16 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">Ready to Experience the Future of Blood Group Testing?</h2>
        <p className="mb-6">Join the growing number of medical facilities using our innovative technology.</p>
        <button className="bg-white text-indigo-600 font-semibold px-6 py-2 rounded-lg hover:bg-gray-100 transition-colors">
          Get Started
        </button>
      </div>
    </div>
  );
}

export default Services;