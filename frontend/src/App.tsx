import './App.css'
import './index.css'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-starbucks-cream to-starbucks-light text-starbucks-brown font-sans flex flex-col">
      {/* Header */}
      <header className="bg-starbucks-green text-white px-6 py-4 shadow-lg flex justify-between items-center">
        <h1 className="text-2xl font-bold">Starbucks Gym Checker 🏋️‍♀️</h1>
      </header>

      {/* Main Content */}
      <main className="flex-grow flex items-center justify-center py-8 px-4">
        <div className="w-full max-w-xl bg-white rounded-xl shadow-lg p-8 space-y-6">
          <h2 className="text-2xl font-semibold text-center text-starbucks-green">
            Welcome to the Checker!
          </h2>
          <p className="text-center text-gray-600">
            Find out if your drink is gym-approved or a cheat-day treat.
          </p>

          {/* Form starts here */}
          <form className="space-y-4">
            {/* Drink Type */}
              <div className="flex items-center gap-4">
                <label htmlFor="drinkType" className="w-1/3 text-sm font-medium text-gray-700">
                  Drink Type
                </label>
                <select
                  id="drinkType"
                  className="w-2/3 border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-starbucks-green"
                >
                  <option value="">Please select</option>
                  <option value="coffee">Coffee</option>
                  <option value="latte">Latte</option>
                  <option value="frappuccino">Frappuccino</option>
                  <option value="espresso">Espresso</option>
                </select>
              </div>

              {/* Size */}
              <div className="flex items-center gap-4">
                <label htmlFor="drinkSize" className="w-1/3 text-sm font-medium text-gray-700">
                  Size
                </label>
                <select
                  id="drinkSize"
                  className="w-2/3 border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-starbucks-green"
                >
                  <option value="">Please select</option>
                  <option value="short">Short</option>
                  <option value="tall">Tall</option>
                  <option value="grande">Grande</option>
                  <option value="venti">Venti</option>
                </select>
              </div>

              {/* Milk Type */}
              <div className="flex items-center gap-4">
                <label htmlFor="milkType" className="w-1/3 text-sm font-medium text-gray-700">
                  Milk Type
                </label>
                <select
                  id="milkType"
                  className="w-2/3 border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-starbucks-green"
                >
                  <option value="">Please select</option>
                  <option value="regular">Regular</option>
                  <option value="soy">Soy</option>
                  <option value="almond">Almond</option>
                  <option value="oat">Oat</option>
                </select>
              </div>

              {/* Whipped Cream */}
              <div className="flex items-center gap-4">
                <span className="w-1/3 text-sm font-medium text-gray-700">
                  Whipped Cream
                </span>
                <div className="w-2/3 flex items-center space-x-6">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="whip"
                      value="with"
                      className="form-radio text-starbucks-green focus:ring-starbucks-green"
                    />
                    <span className="ml-2 text-gray-700">Yes</span>
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="whip"
                      value="without"
                      className="form-radio text-starbucks-green focus:ring-starbucks-green"
                    />
                    <span className="ml-2 text-gray-700">No</span>
                  </label>
                </div>
              </div>


            {/* Submit Button */}
            <div className="flex justify-center pt-4">
              <button
                type="submit"
                className="px-6 py-2 bg-starbucks-green text-white rounded-full hover:bg-starbucks-highlight transition duration-300"
              >
                Check Your Drink
              </button>
            </div>
          </form>
          {/* Form ends here */}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 text-center text-sm text-gray-600 py-4">
        © {new Date().getFullYear()} Starbucks Gym Checker. All rights reserved.
      </footer>
    </div>
  )
}

export default App
