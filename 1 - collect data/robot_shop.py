import os
import random

from locust import HttpUser, task, between

TIMEOUT_FAST = 10
TIMEOUT_SLOW = 60


class ShopUser(HttpUser):

    wait_time = between(2, 10)
    _FAKE_IPS = [
        "156.33.241.5",
        "34.196.93.245",
        "98.142.103.241",
        "192.241.230.151",
        "46.114.35.116",
        "52.77.99.130",
        "60.242.161.215",
    ]

    # Helpers
    def _ip(self) -> dict:
        """Return an HTTP header dict with a random forwarded IP."""
        return {"x-forwarded-for": random.choice(self._FAKE_IPS)}

    def _json(self, response) -> dict | list | None:
        try:
            if response is None or response.status_code != 200:
                return None
            text = response.text
            if not text or not text.strip():
                return None
            return response.json()
        except Exception:
            return None

    # Tasks

    @task(1)
    def login(self) -> None:
        """Authenticate with the user service."""
        self.client.post(
            "/api/user/login",
            json={"name": "user", "password": "password"},
            headers=self._ip(),
            timeout=TIMEOUT_FAST,
        )

    @task(5)
    def load(self) -> None:
        """
        Execute the full purchase funnel.

        Steps
        -----
        1. Fetch a unique session ID from the user service.
        2. List catalogue categories and products.
        3. Rate and add two in-stock items to the cart.
        4. Select a shipping destination and confirm the shipment.
        5. Submit payment.

        Any step that returns an invalid response aborts the task
        gracefully so that subsequent tasks are not affected.
        """
        ip = self._ip()

        # Home page
        self.client.get("/", headers=ip, timeout=TIMEOUT_FAST)

        # --- User service ---
        user_resp  = self.client.get("/api/user/uniqueid", headers=ip, timeout=TIMEOUT_FAST)
        user       = self._json(user_resp)
        if not user or "uuid" not in user:
            return
        session_id = user["uuid"]

        # --- Catalogue ---
        self.client.get("/api/catalogue/categories", headers=ip, timeout=TIMEOUT_FAST)
        products_resp = self.client.get("/api/catalogue/products", headers=ip, timeout=TIMEOUT_FAST)
        products      = self._json(products_resp)
        if not products:
            return

        in_stock = [p for p in products if p.get("instock", 0) != 0]
        if not in_stock:
            return

        # Add two items to the cart
        for _ in range(2):
            item = random.choice(in_stock)

            # Optionally submit a rating (30% probability)
            if random.randint(1, 10) <= 3:
                self.client.put(
                    f"/api/ratings/api/rate/{item['sku']}/{random.randint(1, 5)}",
                    headers=ip,
                    timeout=TIMEOUT_FAST,
                )

            self.client.get(f"/api/catalogue/product/{item['sku']}", headers=ip, timeout=TIMEOUT_FAST)
            self.client.get(f"/api/ratings/api/fetch/{item['sku']}", headers=ip, timeout=TIMEOUT_FAST)
            self.client.get(f"/api/cart/add/{session_id}/{item['sku']}/1", headers=ip, timeout=TIMEOUT_FAST)

        # --- Cart ---
        cart_resp = self.client.get(f"/api/cart/cart/{session_id}", headers=ip, timeout=TIMEOUT_FAST)
        cart      = self._json(cart_resp)
        if not cart or not cart.get("items"):
            return

        item = random.choice(cart["items"])
        self.client.get(
            f"/api/cart/update/{session_id}/{item['sku']}/2", headers=ip, timeout=TIMEOUT_FAST
        )

        # --- Shipping ---
        codes_resp = self.client.get("/api/shipping/codes", headers=ip, timeout=TIMEOUT_SLOW)
        codes      = self._json(codes_resp)
        if not codes:
            return
        code = random.choice(codes)

        cities_resp = self.client.get(
            f"/api/shipping/cities/{code['code']}", headers=ip, timeout=TIMEOUT_SLOW
        )
        cities = self._json(cities_resp)
        if not cities:
            return
        city = random.choice(cities)

        shipping_resp = self.client.get(
            f"/api/shipping/calc/{city['uuid']}", headers=ip, timeout=TIMEOUT_SLOW
        )
        shipping = self._json(shipping_resp)
        if not shipping:
            return
        shipping["location"] = f"{code['name']} {city['name']}"

        confirm_resp = self.client.post(
            f"/api/shipping/confirm/{session_id}", json=shipping, headers=ip, timeout=TIMEOUT_SLOW
        )
        cart = self._json(confirm_resp)
        if not cart:
            return

        # --- Payment ---
        self.client.post(
            f"/api/payment/pay/{session_id}", json=cart, headers=ip, timeout=TIMEOUT_FAST
        )

    @task(1)
    def error(self) -> None:
        """
        Trigger a payment error if the ERROR environment variable is set.

        Used to deliberately inject failures during chaos/resilience testing
        without modifying the main load task.
        """
        if os.environ.get("ERROR") == "1":
            self.client.post(
                "/api/payment/pay/partner-57",
                json={"total": 0, "tax": 0},
                headers=self._ip(),
            )