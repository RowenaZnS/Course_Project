import random
import re
from datetime import datetime, timedelta

class Assistant:
    def __init__(self):
        # Mapping of keywords to handler functions
        self.intent_handlers = {
            'weather': self.handle_weather,
            'forecast': self.handle_weather,
            'restaurant': self.handle_restaurant,
            'food': self.handle_restaurant,
            'eat': self.handle_restaurant,
            'bus': self.handle_transport,
            'tram': self.handle_transport,
            'transport': self.handle_transport,
        }

        # Default responses when intent is unclear
        self.default_responses = [
            "I'm sorry, I didn't quite understand that. Can you rephrase?",
            "Could you please clarify your request?",
            "I'm not sure how to help with that. Maybe try asking about weather, restaurants, or transport?"
        ]

        # Weather data for cities with random initialization
        self.weather_cities = [
            "gothenburg", "copenhagen", "barcelona", "beijing",
            "los angeles", "stockholm", "tokyo", "paris",
            "new york", "sydney"
        ]

        # Month mapping for parsing
        self.month_mapping = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

    def generate_random_weather(self):
        """Generate random weather conditions and temperature."""
        condition = random.choice(["sunny", "cloudy", "rainy", "snowy", "windy", "smoggy", "stormy"])
        temperature = random.randint(-10, 40)  # Random temperature between -10°C and 40°C
        return {"condition": condition, "temperature": temperature}

    def parse_date(self, user_input):
        """Extract month and day from user input in 'Month Day' format."""
        # Match month and day (e.g., March 15)
        date_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b\s+(\d{1,2})', user_input, re.IGNORECASE)
        if date_match:
            month_str = date_match.group(1).lower()  # Extract month name
            day = int(date_match.group(2))  # Extract day
            month = self.month_mapping.get(month_str)  # Convert month name to number
            return month, day
        return None, None

    def handle_weather(self, user_input):
        # Identify city from user input (case insensitive)
        user_input_lower = user_input.lower()
        selected_city = None
        for city in self.weather_cities:
            if city in user_input_lower:
                selected_city = city
                break

        if not selected_city:
            return "I couldn't find weather information for that city in my database. Please specify a valid city."

        # Parse date from user input
        month, day = self.parse_date(user_input)
        if month and day:
            try:
                # Validate the date
                specified_date = datetime(datetime.now().year, month, day)
                specific_weather = self.generate_random_weather()
                return f"The weather in {selected_city.capitalize()} on {specified_date.strftime('%B %d')} is expected to be {specific_weather['condition']} with a temperature of {specific_weather['temperature']}°C."
            except ValueError:
                return "The date you provided is invalid. Please provide a valid date (e.g., March 15)."

        # If no specific date is provided, return today's weather
        today_weather = self.generate_random_weather()
        return f"The weather in {selected_city.capitalize()} today is {today_weather['condition']} with a temperature of {today_weather['temperature']}°C."

    def handle_restaurant(self, user_input):
        # Restaurant data for regions
        restaurant_data = {
            "swedish": ["Smorgasbord Paradise", "Swedish Meatball House", "Nordic Kitchen"],
            "american": ["Burger Town", "Steakhouse Grill", "Fried Chicken Heaven"],
            "russian": ["Borscht Bistro", "Caviar Dreams", "Moscow Delights"],
            "chinese": ["Dim Sum Express", "Peking Duck Palace", "Hotpot Heaven"],
        }

        # Identify region from user input (case insensitive)
        user_input_lower = user_input.lower()
        for region in restaurant_data.keys():
            if region in user_input_lower:
                restaurants = restaurant_data[region]
                chosen_restaurant = random.choice(restaurants)
                return f"In {region.capitalize()}, I recommend '{chosen_restaurant}', it's highly rated!"
        # If region not specified
        regions = list(restaurant_data.keys())
        return f"I couldn't find restaurant recommendations for that specific area. Please specify a region, such as {', '.join(regions)}."

    def handle_transport(self, user_input):
        # Transport data for stations
        transport_data = {
            "central station": {"tram": random.randint(1, 10), "bus": random.randint(1, 10)},
            "chalmers station": {"tram": random.randint(1, 10), "bus": random.randint(1, 10)},
            "haga station": {"tram": random.randint(1, 10), "bus": random.randint(1, 10)},
            "majorna station": {"tram": random.randint(1, 10), "bus": random.randint(1, 10)},
            "lindholmen station": {"tram": random.randint(1, 10), "bus": random.randint(1, 10)},
        }

        # Identify station and transport type from user input (case insensitive)
        user_input_lower = user_input.lower()
        transport_type = None

        if "bus" in user_input_lower:
            transport_type = "bus"
        elif "tram" in user_input_lower:
            transport_type = "tram"

        for station in transport_data.keys():
            if station in user_input_lower:
                if transport_type:
                    next_arrival = random.randint(1, 15)
                    arrival_time = (datetime.now() + timedelta(minutes=next_arrival)).strftime("%H:%M")
                    return f"The next {transport_type} at {station.capitalize()} arrives at {arrival_time}."
                else:
                    transport = random.choice(list(transport_data[station].keys()))
                    next_arrival = random.randint(1, 15)
                    arrival_time = (datetime.now() + timedelta(minutes=next_arrival)).strftime("%H:%M")
                    return f"The next {transport} at {station.capitalize()} arrives at {arrival_time}."
        return "I couldn't find transport information for that station. Please specify a valid station."

    def detect_intent(self, user_input):
        # Convert input to lowercase and tokenize
        words = re.findall(r'\w+', user_input.lower())

        for word in words:
            if word in self.intent_handlers:
                return self.intent_handlers[word]
        return None

    def get_response(self, user_input):
        handler = self.detect_intent(user_input)
        if handler:
            return handler(user_input)
        else:
            return random.choice(self.default_responses)

    def chat(self):
        print("Assistant: Hi! How can I help you today? (Type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Assistant: Goodbye! Have a great day!")
                break
            response = self.get_response(user_input)
            print(f"Assistant: {response}")

if __name__ == "__main__":
    assistant = Assistant()
    assistant.chat()