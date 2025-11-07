import { initXGBoostCtor } from "../xgb";

declare global {
  // For TypeScript to accept our test-time assignment
  // eslint-disable-next-line no-var
  var fetch: typeof fetch;
}

describe("initXGBoostCtor", () => {
  beforeAll(() => {
    // Minimal fetch stub that returns a valid ArrayBuffer response.
    global.fetch = jest.fn(async () => {
      const buf = new ArrayBuffer(8);
      return new Response(buf, {
        status: 200,
        headers: { "Content-Type": "application/wasm" },
      }) as any;
    }) as any;

    // Ensure self is defined (jsdom env provides window; align self with it)
    if (typeof self === "undefined" && typeof window !== "undefined") {
      // @ts-ignore
      global.self = window;
    }
  });

  it("returns a usable XGBoost class from mocked ml-xgboost", async () => {
    const XGBoost = await initXGBoostCtor();
    const model = new XGBoost({});

    expect(typeof model.train).toBe("function");
    expect(typeof model.predict).toBe("function");

    model.train([[1, 2]], [3]);
    const preds = model.predict([[1, 2]]);
    expect(Array.isArray(preds)).toBe(true);
  });
});
