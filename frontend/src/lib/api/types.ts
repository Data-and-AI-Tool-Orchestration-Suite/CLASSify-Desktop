/**
 * Placeholder type definitions.
 *
 * Replace with generated types by running:
 *   npm run gen-types   (requires the backend running on :8000)
 *
 * This file is git-ignored when generated; the placeholder is committed
 * only so `svelte-check` and `npm test` pass before the first generation.
 */
export interface paths {
  "/api/system/health": {
    get: { responses: { 200: { content: { "application/json": { status: string } } } } };
  };
  "/api/system/info": {
    get: {
      responses: {
        200: {
          content: {
            "application/json": {
              app: string;
              version: string;
              os: string;
              os_version: string;
              arch: string;
              python: string;
              data_dir: string;
              dev_mode: boolean;
            };
          };
        };
      };
    };
  };
}
