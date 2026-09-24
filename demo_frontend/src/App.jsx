import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { ScanSearch, CarFront, Loader2, UploadCloud, ArrowRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const API_URL = "http://localhost:8000/api";

function App() {
  const [modelName, setModelName] = useState("resnet_finetuned");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const selected = acceptedFiles[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResults([]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpeg", ".jpg", ".png"] },
    multiple: false,
  });

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", modelName);

    try {
      const response = await axios.post(`${API_URL}/search/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data.error) {
        setError(response.data.error);
      } else {
        setResults(response.data.results);
      }
    } catch (err) {
      setError("Connection error: Server might be down.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-zinc-100 selection:text-zinc-900 font-sans p-6 md:p-12 transition-colors duration-300 dark">
      {/* Header */}
      <header className="mb-12 flex flex-col md:flex-row md:items-center justify-between border-b border-zinc-800 pb-6">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight flex items-center gap-3 text-white">
            <ScanSearch className="w-10 h-10" />
            Vehicle Re-ID
            <span className="font-light text-zinc-500">AI Demo</span>
          </h1>
          <p className="text-zinc-400 mt-2 text-sm font-medium uppercase tracking-widest">
            Graduation Project
          </p>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-10">
        {/* Left Side: Controls & Upload */}
        <section className="lg:col-span-4 flex flex-col gap-8">
          <Card className="bg-zinc-900/40 border-zinc-800/60 backdrop-blur-xl shadow-2xl">
            <CardHeader>
              <CardTitle className="text-xl font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                Query Configuration
              </CardTitle>
              <CardDescription className="text-zinc-400">
                Configure your search model and upload a target vehicle.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-8">
              
              <div className="flex flex-col gap-3">
                <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                  Model Architecture
                </label>
                <Select value={modelName} onValueChange={setModelName}>
                  <SelectTrigger className="w-full bg-zinc-950 border-zinc-800 text-zinc-100">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent className="bg-zinc-950 border-zinc-800 text-zinc-100">
                    <SelectItem value="resnet_baseline">ResNet-IBN (Baseline)</SelectItem>
                    <SelectItem value="resnet_finetuned">ResNet-IBN (Fine-tuned)</SelectItem>
                    <SelectItem value="swin_baseline">Swin-T Transformer (Baseline)</SelectItem>
                    <SelectItem value="swin_finetuned">Swin-T Transformer (Fine-tuned)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-3">
                <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                  Vehicle Image
                </label>
                <div
                  {...getRootProps()}
                  className={`relative border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 ${
                    isDragActive
                      ? "border-white bg-white/5 scale-[1.02]"
                      : "border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/30"
                  }`}
                >
                  <input {...getInputProps()} />
                  {preview ? (
                    <div className="relative w-full aspect-video rounded-lg overflow-hidden group">
                      <div className="absolute top-2 left-2 z-10 bg-black/80 backdrop-blur-md px-2 py-1 rounded text-xs font-bold font-mono text-white">
                        ID: {file.name.split('_')[0]}
                      </div>
                      <img src={preview} alt="Query preview" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <p className="text-sm font-semibold z-20 relative">Change Image</p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center mb-4">
                        <UploadCloud className="w-8 h-8 text-zinc-300" />
                      </div>
                      <p className="text-zinc-300 font-medium">Drag & drop image</p>
                      <p className="text-zinc-500 text-sm mt-1">or click to browse</p>
                    </>
                  )}
                </div>
              </div>

              <Button
                onClick={handleAnalyze}
                disabled={!file || loading}
                variant="default"
                size="lg"
                className="w-full bg-white text-black hover:bg-zinc-200 group"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    Search Gallery
                    <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </Button>

              {error && (
                <div className="mt-4 bg-red-950/50 border border-red-900/50 text-red-200 text-sm p-4 rounded-xl flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 flex-shrink-0"></div>
                  {error}
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Right Side: Results Gallery */}
        <section className="lg:col-span-8">
          <Card className="bg-zinc-900/20 border-zinc-800/40 h-full min-h-[600px] backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-2xl font-bold tracking-tight text-white">Match Results</CardTitle>
              {results.length > 0 && (
                <Badge variant="outline" className="text-zinc-300 border-zinc-700 bg-zinc-800/50">
                  Top {results.length} Matches
                </Badge>
              )}
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex flex-col items-center justify-center h-[400px] text-zinc-400">
                  <Loader2 className="w-12 h-12 animate-spin mb-4 text-white" />
                  <p className="animate-pulse">Extracting deep features and computing similarities...</p>
                </div>
              ) : results.length > 0 ? (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                  {results.map((item, index) => {
                    const isMatch = file && item.path.split('_')[0] === file.name.split('_')[0];
                    const borderColor = isMatch ? "border-emerald-500/70 hover:border-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.15)]" : "border-rose-500/70 hover:border-rose-400 shadow-[0_0_15px_rgba(244,63,94,0.15)]";
                    const badgeColor = isMatch ? "bg-emerald-500 text-white" : "bg-rose-500 text-white";
                    
                    return (
                    <div 
                      key={index} 
                      className={`group relative bg-zinc-950 rounded-xl overflow-hidden border-2 transition-all duration-300 hover:-translate-y-1 ${borderColor}`}
                      style={{ animation: `fadeIn 0.5s ease-out ${index * 0.05}s both` }}
                    >
                      <div className="absolute top-2 left-2 z-10 bg-black/80 backdrop-blur-md px-2 py-1 rounded text-xs font-bold font-mono text-white">
                        #{index + 1}
                      </div>
                      <div className={`absolute top-2 right-2 z-10 px-2 py-1 rounded text-xs font-bold ${badgeColor}`}>
                        {item.score}%
                      </div>
                      <div className="absolute bottom-2 left-2 z-10 bg-black/80 backdrop-blur-md px-2 py-1 rounded text-xs font-bold font-mono text-zinc-300">
                        ID: {item.path.split('_')[0]}
                      </div>
                      <div className="aspect-square overflow-hidden bg-zinc-900">
                        <img 
                          src={`${API_URL}/image/${item.path}`} 
                          alt={item.path} 
                          className="w-full h-full object-cover transition-all duration-500 scale-100 group-hover:scale-110" 
                        />
                      </div>
                    </div>
                  )})}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[400px] text-zinc-500 text-center">
                  <div className="w-24 h-24 rounded-full bg-zinc-900/50 flex items-center justify-center mb-6">
                    <CarFront className="w-12 h-12 text-zinc-700" />
                  </div>
                  <h3 className="text-xl font-medium text-zinc-300 mb-2">No results yet</h3>
                  <p className="max-w-md">Upload a vehicle image on the left and click "Search Gallery" to find matches using our deep learning models.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      </main>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}} />
    </div>
  );
}

export default App;
