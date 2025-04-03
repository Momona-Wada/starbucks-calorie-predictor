"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "./components/ui/button"
import { Input } from "./components/ui/input"
import { Coffee, Loader2 } from "lucide-react"
import { Card, CardContent } from "./components/ui/card"
import { Badge } from "./components/ui/badge"

function App() {
  const [userInput, setUserInput] = useState("")
  const [result, setResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  const API_URL = "http://127.0.0.1:8000/recommend_text"

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!userInput) return

    setIsLoading(true)
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userInput }),
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error("Error calling API:", error)
      setResult({ error: "Failed to fetch from API" })
    } finally {
      setIsLoading(false)
    }
  }

  const imageMap: Record<string, string> = {
    "Brewed Coffee": "/images/drinks/brewed_coffee.png",
    "Cappuccino": "/images/drinks/cappuccino.png",
    "Ice Brewed Coffee": "/images/drinks/cold_brew.png",
    "Espresso": "/images/drinks/espresso.png",
    "Frappuccino": "/images/drinks/moca_frappuccino",
    "Caffè Latte": "/images/drinks/caffe_latte.png",
    "White Chocolate Mocha": "/images/drinks/white_chocolate_mocha.png",
    "Caramel Macchiato": "/images/drinks/caramel_macchiato.png",
    "Java Chip": "/images/drinks/java_chip.png",
    "No drinks found at all for these tags": "/images/drinks/no_image_available"
  }

  return (
    <div className="min-h-screen bg-starbucks-light/30">
      <header className="bg-starbucks-green text-white py-6 shadow-md">
        <div className="container mx-auto px-4 flex items-center">
          <Coffee className="h-8 w-8 mr-3" />
          <h1 className="text-2xl font-bold">Starbucks Drink Finder</h1>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Card className="max-w-3xl mx-auto bg-white shadow-lg border-starbucks-mint/30">
          <CardContent className="p-6">
            <p className="text-gray-600 mb-6">
              Describe what you're in the mood for, and we'll recommend the perfect Starbucks drink for you.
            </p>

            <form onSubmit={handleSubmit} className="flex gap-2 mb-8">
              <Input
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder="I want something sweet and refreshing..."
                className="flex-grow border-starbucks-mint focus-visible:ring-starbucks-green"
              />
              <Button
                type="submit"
                className="bg-starbucks-green hover:bg-starbucks-green/90 text-white"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Finding...
                  </>
                ) : (
                  "Get Recommendations"
                )}
              </Button>
            </form>

            {result && (
              <div className="mt-6">
                {result.error && <div className="p-4 bg-red-50 text-red-700 rounded-md">Error: {result.error}</div>}

                {!result.error && (
                  <>
                    <div className="mb-4 p-4 bg-starbucks-light rounded-md">
                      <p className="font-medium text-starbucks-green">{result.message}</p>
                    </div>

                    {result.recommendations && result.recommendations.length > 0 ? (
                      <div className="grid gap-4 md:grid-cols-2">
                        {result.recommendations.map((item: any, index: number) => (
                          <Card
                            key={index}
                            className="overflow-hidden border-starbucks-mint/20 hover:shadow-md transition-shadow"
                          >
                            <img
                              src={imageMap[item.Beverage] ?? "/images/drinks/no_image_available.png"}
                                alt={item.Beverage}
                                className="w-full h-40 object-contain bg-white p-2"
                            />
                            <div className="bg-starbucks-mint/20 p-3 border-b border-starbucks-mint/20">
                              <h3 className="font-bold text-starbucks-green">{item.Beverage}</h3>
                            </div>
                            <CardContent className="p-4">
                              <div className="flex flex-wrap gap-2 mb-2">
                                {item.tags &&
                                  Array.isArray(item.tags) &&
                                  item.tags.map((tag: string, tagIndex: number) => (
                                    <Badge
                                      key={tagIndex}
                                      variant="outline"
                                      className="bg-starbucks-cream border-starbucks-mint text-starbucks-brown"
                                    >
                                      {tag}
                                    </Badge>
                                  ))}
                              </div>
                              {item.score !== undefined && (
                                <div className="mt-2 text-sm text-gray-500">
                                  Match score:{" "}
                                  <span className="font-medium text-starbucks-green">{item.score.toFixed(2)}</span>
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <p className="text-center py-4 text-gray-500">
                        No recommendations found. Try a different description.
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      <footer className="mt-auto py-4 text-center text-sm text-gray-500">
        <p>Powered by AI • Find your perfect Starbucks drink</p>
      </footer>
    </div>
  )
}

export default App

