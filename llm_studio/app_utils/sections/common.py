import hashlib
import logging

from h2o_wave import Q, ui

from llm_studio.app_utils.cards import card_zones
from llm_studio.app_utils.config import default_cfg

logger = logging.getLogger(__name__)


MATELIX_WAVE_THEME_PATCH_JS = """
function patchMatelixWaveThemeColors() {
  const target = "#E2E8F0";
  const blueTarget = "#38BDF8";

  const waveVars = [
    "--wave-primary",
    "--wave-primary0",
    "--wave-primary1",
    "--wave-primary2",
    "--wave-primary3",
    "--wave-primary4",
    "--wave-primary5",
    "--wave-primary6",
    "--wave-primary7",
    "--wave-primary8",
    "--wave-primary9",

    "--wave-themePrimary",
    "--wave-themeSecondary",
    "--wave-themeTertiary",
    "--wave-themeDark",
    "--wave-themeDarkAlt",
    "--wave-themeDarker",
    "--wave-themeLight",
    "--wave-themeLighter",
    "--wave-themeLighterAlt",

    "--wave-yellow",
    "--wave-amber",
    "--wave-lime"
  ];

  waveVars.forEach((name) => {
    document.documentElement.style.setProperty(name, target);
    if (document.body) {
      document.body.style.setProperty(name, target);
    }
  });

  // Override generated Wave / Fluent classes like .link-235 and .root-263.
  document.querySelectorAll(
    '[class^="link-"], [class*=" link-"], [class^="root-"], [class*=" root-"]'
  ).forEach((el) => {
    el.style.setProperty("color", target, "important");
    el.style.setProperty("text-decoration-color", target, "important");
  });

  document.querySelectorAll(
    '[class^="link-"] *, [class*=" link-"] *, [class^="root-"] *, [class*=" root-"] *'
  ).forEach((el) => {
    el.style.setProperty("color", target, "important");
    el.style.setProperty("fill", target, "important");
    el.style.setProperty("stroke", target, "important");
    el.style.setProperty("text-decoration-color", target, "important");
  });

  // Override generated Wave / Fluent pill classes like .pill-303.
  // These should use the MaTeLiX light blue accent, not the gray fallback.
  document.querySelectorAll(
    '[class^="pill-"], [class*=" pill-"]'
  ).forEach((el) => {
    el.style.setProperty("background", blueTarget, "important");
    el.style.setProperty("background-color", blueTarget, "important");
    el.style.setProperty("border-color", blueTarget, "important");
  });

  // Override slider thumbs created by Fluent/Wave.
  document.querySelectorAll(
    '.ms-Slider-thumb'
  ).forEach((el) => {
    el.style.setProperty("border-color", blueTarget, "important");
  });

  // Fallback: catch inline styles created after Wave rendering.
  document.querySelectorAll("*").forEach((el) => {
    const style = el.getAttribute("style") || "";
    const styleLower = style.toLowerCase();

    const hasWaveYellow =
      style.includes("rgb(254, 201, 37)") ||
      style.includes("rgb(254,201,37)") ||
      style.includes("rgb(254 201 37)") ||
      style.includes("rgb(194, 153, 29)") ||
      style.includes("rgb(194,153,29)") ||
      style.includes("rgb(194 153 29)") ||
      styleLower.includes("#fec925") ||
      styleLower.includes("fec925") ||
      styleLower.includes("#ffcf40") ||
      styleLower.includes("ffcf40") ||
      styleLower.includes("#ffde7d") ||
      styleLower.includes("ffde7d") ||
      styleLower.includes("#c2991d") ||
      styleLower.includes("c2991d");

    if (hasWaveYellow) {
      el.style.setProperty("color", target, "important");
      el.style.setProperty("fill", target, "important");
      el.style.setProperty("stroke", target, "important");
      el.style.setProperty("border-color", blueTarget, "important");
      el.style.setProperty("text-decoration-color", target, "important");
    }
  });
}

function startMatelixWaveThemePatch() {
  patchMatelixWaveThemeColors();

  // Wave applies theme values asynchronously, so patch several times.
  setTimeout(patchMatelixWaveThemeColors, 50);
  setTimeout(patchMatelixWaveThemeColors, 250);
  setTimeout(patchMatelixWaveThemeColors, 1000);
  setTimeout(patchMatelixWaveThemeColors, 2000);

  if (!window.__matelixWaveThemeObserverStarted && document.body) {
    window.__matelixWaveThemeObserverStarted = true;

    new MutationObserver(() => {
      patchMatelixWaveThemeColors();
    }).observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["style", "class"]
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", startMatelixWaveThemePatch);
} else {
  startMatelixWaveThemePatch();
}
"""


def matelix_wave_theme_patch() -> ui.InlineScript:
    return ui.inline_script(content=MATELIX_WAVE_THEME_PATCH_JS)


async def meta(q: Q) -> None:
    if q.client["keep_meta"]:  # Do not reset meta, keep current dialog opened
        q.client["keep_meta"] = False
        return

    zones = card_zones(mode=q.client["mode_curr"])

    if q.client["notification_bar"]:
        notification_bar = ui.notification_bar(
            type="warning",
            timeout=20,
            text=q.client["notification_bar"],
            position="top-right",
        )
    else:
        notification_bar = None

    # TODO remove `stylesheet` when wave makes message bars smaller
    q.page["meta"] = ui.meta_card(
        box="",
        title="MaTeLiX AI Studio",
        icon="https://matelix.ai/pic/SVG.png",
        layouts=[
            ui.layout(breakpoint="0px", width="100%", zones=zones),
            ui.layout(breakpoint="1920px", width="1920px", zones=zones),
        ],
        scripts=[
            ui.script(source, asynchronous=True) for source in q.app["script_sources"]
        ],
        stylesheet=ui.inline_stylesheet(
            """
            .ms-MessageBar {
              padding-top: 3px;
              padding-bottom: 3px;
              min-height: 18px;
            }

            div[data-test="nav_bar"] .ms-Nav-groupContent {
              margin-bottom: 0;
            }

            div[data-test="experiment/display/deployment/top_right"],
            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div > div {
                display: flex;
            }

            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child,
            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div {
                display: flex;
                flex-grow: 1;
            }

            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div > div > div {
                display: flex;
                flex-grow: 1;
                flex-direction: column;
            }

            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div > div > div > div {
                flex-grow: 1;
            }

            div[data-test] .ms-Button,
            div[data-test] .ms-TextField-fieldGroup,
            div[data-test] .ms-Dropdown-title,
            div[data-test] .ms-DetailsHeader,
            div[data-test] .ms-DetailsRow {
                border-radius: 10px;
            }

            div[data-test] .ms-Button--primary {
                background: linear-gradient(120deg, #38BDF8 0%, #0EA5E9 100%);
                border: none;
            }

            div[data-test="header"] {
                --wave-text: #FFFFFF;
                --wave-card: #FFFFFF;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
                background: linear-gradient(135deg, #111827 0%, #1E293B 55%, #7C4DFF 100%) !important;
            }

            div[data-test="header"] .ms-Card {
                background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 41, 59, 0.92) 55%, rgba(124, 77, 255, 0.88) 100%);
                border: 1px solid rgba(255, 255, 255, 0.14);
                backdrop-filter: blur(6px);
            }

            div[data-test="header"] .ms-Card-title {
                font-size: 1.25rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                color: #FFFFFF !important;
            }

            div[data-test="header"] .wave-s24.wave-w5,
            div[data-test="header"] .wave-w5 {
                color: #FFFFFF !important;
            }

            div[data-test="header"] .ffxq3ur {
                color: #FFFFFF !important;
            }

            div[data-test="header"] .ms-Card-subtitle,
            div[data-test="header"] .ms-Card-subtitle *,
            div[data-test="header"] [class*="subtitle"],
            div[data-test="header"] [class*="Subtitle"] {
                opacity: 0.95;
                font-weight: 500;
                color: #FFFFFF !important;
                fill: #FFFFFF !important;
            }

            /* Enforce MaTeLiX accent colors via theme tokens */
            :root,
            body {
                --themePrimary: #38BDF8;
                --themeLighterAlt: #F5F3FF;
                --themeLighter: #EDE9FE;
                --themeLight: #DDD6FE;
                --themeTertiary: #7DD3FC;
                --themeSecondary: #38BDF8;
                --themeDarkAlt: #0284C7;
                --themeDark: #0369A1;
                --themeDarker: #0C4A6E;
                --paletteYellow: #E2E8F0;

                --wave-primary: #E2E8F0 !important;
                --wave-themePrimary: #E2E8F0 !important;
                --wave-themeSecondary: #E2E8F0 !important;
                --wave-themeTertiary: #E2E8F0 !important;
                --wave-themeDark: #E2E8F0 !important;
                --wave-themeDarkAlt: #E2E8F0 !important;
                --wave-themeDarker: #E2E8F0 !important;
                --wave-yellow: #E2E8F0 !important;
                --wave-amber: #E2E8F0 !important;
                --wave-lime: #E2E8F0 !important;
            }

            /* Override Wave-generated Fluent/Wave classes like .link-235 and .root-263 */
            [class^="link-"],
            [class*=" link-"],
            [class^="root-"],
            [class*=" root-"] {
                color: #E2E8F0 !important;
                text-decoration-color: #E2E8F0 !important;
            }

            [class^="link-"] *,
            [class*=" link-"] *,
            [class^="root-"] *,
            [class*=" root-"] * {
                color: #E2E8F0 !important;
                fill: #E2E8F0 !important;
                stroke: #E2E8F0 !important;
                text-decoration-color: #E2E8F0 !important;
            }

            /* Override Wave-generated pseudo elements like .link-232::after */
            [class^="link-"]::before,
            [class*=" link-"]::before,
            [class^="link-"]::after,
            [class*=" link-"]::after {
                border-left-color: #38BDF8 !important;
                border-right-color: #38BDF8 !important;
                border-top-color: #38BDF8 !important;
                border-bottom-color: #38BDF8 !important;
            }

            /* Override selected link indicators like .linkIsSelected-457::before */
            [class^="linkIsSelected-"]::before,
            [class*=" linkIsSelected-"]::before,
            [class^="linkIsSelected-"]::after,
            [class*=" linkIsSelected-"]::after {
                background: #38BDF8 !important;
                background-color: #38BDF8 !important;
                border-color: #38BDF8 !important;
            }

            /* Override Wave-generated dropdown focus rings like .dropdown-273:focus::after */
            [class^="dropdown-"]:focus::before,
            [class*=" dropdown-"]:focus::before,
            [class^="dropdown-"]:focus::after,
            [class*=" dropdown-"]:focus::after,
            [class^="dropdown-"]:focus-within::before,
            [class*=" dropdown-"]:focus-within::before,
            [class^="dropdown-"]:focus-within::after,
            [class*=" dropdown-"]:focus-within::after {
                border-color: #38BDF8 !important;
            }

            /* Override Wave-generated toggle/pill classes like .pill-303 */
            [class^="pill-"],
            [class*=" pill-"] {
                background: #38BDF8 !important;
                background-color: #38BDF8 !important;
                border-color: #38BDF8 !important;
            }

            /* Override Wave-generated slider classes like .slideBox-389:hover .ms-Slider-thumb */
            [class^="slideBox-"] .ms-Slider-thumb,
            [class*=" slideBox-"] .ms-Slider-thumb,
            [class^="slideBox-"]:hover .ms-Slider-thumb,
            [class*=" slideBox-"]:hover .ms-Slider-thumb {
                border-color: #38BDF8 !important;
            }

            /* Fallback for inline yellow values coming from Wave internals */
            [style*="254, 201, 37"],
            [style*="254,201,37"],
            [style*="254 201 37"],
            [style*="194, 153, 29"],
            [style*="194,153,29"],
            [style*="194 153 29"],
            [style*="FEC925"],
            [style*="fec925"],
            [style*="ffcf40"],
            [style*="FFCF40"],
            [style*="ffde7d"],
            [style*="FFDE7D"],
            [style*="c2991d"],
            [style*="C2991D"] {
              color: #E2E8F0 !important;
              fill: #E2E8F0 !important;
              stroke: #E2E8F0 !important;
              border-color: #38BDF8 !important;
              text-decoration-color: #E2E8F0 !important;
            }
            """
        ),
        script=matelix_wave_theme_patch(),
        notification_bar=notification_bar,
    )

    q.page["meta"].theme = "h2o-dark"


def heap_analytics(
    userid, user_properties=None, event_properties=None
) -> ui.InlineScript:
    script = (
        "window.heap=window.heap||[],heap.load=function(e,t)"
        "{window.heap.appid=e,window.heap."
        'config=t=t||{};var r=document.createElement("script");'
        'r.type="text/javascript",'
        'r.async=!0,r.src="https://cdn.heapanalytics.com/js/heap-"+e+".js";'
        'var a=document.getElementsByTagName("script")[0];'
        "a.parentNode.insertBefore(r,a);"
        "for(var n=function(e){return function(){heap.push([e]."
        "concat(Array.prototype.slice.call(arguments,0)))}},"
        'p=["addEventProperties","addUserProperties","clearEventProperties","identify",'
        '"resetIdentity","removeEventProperty","setEventProperties","track",'
        '"unsetEventProperty"],o=0;o<p.length;o++)heap[p[o]]=n(p[o])};'
        'heap.load("1090178399");'
    )

    identity = hashlib.sha256(userid.encode()).hexdigest()
    script += f"heap.identify('{identity}');"

    if user_properties is not None:
        script += f"heap.addUserProperties({user_properties})"

    if event_properties is not None:
        script += f"heap.addEventProperties({event_properties})"

    script += MATELIX_WAVE_THEME_PATCH_JS

    return ui.inline_script(content=script)


async def interface(q: Q) -> None:
    """Display interface cards."""

    await meta(q)

    navigation_pages = ["Home", "Settings"]

    if q.client["init_interface"] is None:
        # to avoid flickering
        q.page["header"] = ui.header_card(
            box="header",
            title=default_cfg.name,
            image=q.app["icon_path"],
            subtitle="v14.4.MaTeLiX-DEV",
        )

        if q.app.heap_mode:
            logger.info("Heap on")
            q.page["meta"].script = heap_analytics(
                userid=q.auth.subject,
                event_properties=(
                    f"{{version: '{q.app.version}'" + f", product: '{q.app.name}'}}"
                ),
            )
            # execute the heap inline script once in the initialization
            await q.page.save()
        else:
            logger.info("Heap off")

        q.page["nav_bar"] = ui.nav_card(
            box="nav",
            items=[
                ui.nav_group(
                    "Navigation",
                    items=[
                        ui.nav_item(page.lower(), page) for page in navigation_pages
                    ],
                ),
                ui.nav_group(
                    "Datasets",
                    items=[
                        ui.nav_item(name="dataset/import", label="Import dataset"),
                        ui.nav_item(name="dataset/list", label="View datasets"),
                    ],
                ),
                ui.nav_group(
                    "Experiments",
                    items=[
                        ui.nav_item(name="experiment/start", label="Create experiment"),
                        ui.nav_item(
                            name="experiment/create_model", label="Create model"
                        ),
                        ui.nav_item(
                            name="experiment/start/grid_search",
                            label="Create grid search",
                        ),
                        ui.nav_item(name="experiment/list", label="View experiments"),
                    ],
                ),
            ],
            value=(
                default_cfg.start_page
                if q.client["nav/active"] is None
                else q.client["nav/active"]
            ),
        )
    else:
        # Only update menu properties to prevent from flickering
        q.page["nav_bar"].value = (
            default_cfg.start_page
            if q.client["nav/active"] is None
            else q.client["nav/active"]
        )

    q.client["init_interface"] = True


async def clean_dashboard(q: Q, mode: str = "full", exclude: list[str] = []):
    """Drop cards from Q page."""

    logger.info(q.client.delete_cards)
    for card_name in q.client.delete_cards:
        if card_name not in exclude:
            del q.page[card_name]

    q.page["meta"].layouts[0].zones = card_zones(mode=mode)
    q.client["mode_curr"] = mode
    q.client["notification_bar"] = None


async def delete_dialog(q: Q, names: list[str], action, entity):
    title = "Do you really want to delete "
    n_datasets = len(names)

    if n_datasets == 1:
        title = f"{title} {entity} {names[0]}?"
    else:
        title = f"{title} {n_datasets} {entity}s?"

    q.page["meta"].dialog = ui.dialog(
        f"Delete {entity}",
        items=[
            ui.text(title),
            ui.markup("<br>"),
            ui.buttons(
                [
                    ui.button(name=action, label="Delete", primary=True),
                    ui.button(name="abort", label="Abort", primary=False),
                ],
                justify="end",
            ),
        ],
    )
    q.client["keep_meta"] = True


async def info_dialog(q: Q, title: str, message: str):
    q.page["meta"].dialog = ui.dialog(
        title,
        items=[
            ui.text(message),
            ui.markup("<br>"),
            ui.buttons(
                [
                    ui.button(name="abort", label="Continue", primary=False),
                ],
                justify="end",
            ),
        ],
        blocking=True,
    )
    q.client["keep_meta"] = True


async def heap_redact(q: Q) -> None:
    if q.app.heap_mode:
        # Send the page to the browser, so the following js can be applied
        await q.page.save()

        # replace dataset names with ****
        q.page["meta"].script = ui.inline_script(
            content="""
document.querySelectorAll('div[data-automation-key="name"]').forEach(a => {
  a.setAttribute('data-heap-redact-text', '')
})

const datasetsTable = document.querySelector(
  'div[data-test="datasets_table"] .ms-ScrollablePane--contentContainer'
);

if (datasetsTable) {
  datasetsTable.addEventListener('scroll', () => {
    window.setTimeout(() => {
      document.querySelectorAll('div[data-automation-key="name"]').forEach(a => {
        a.setAttribute('data-heap-redact-text', '')
      })
    }, 100)
  })
}
"""
            + MATELIX_WAVE_THEME_PATCH_JS
        )
