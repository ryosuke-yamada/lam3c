import { Link } from "react-router";
import { Menu } from "lucide-react";

import { Button } from "./ui/button";
import { ThemeToggle } from "./ui/theme-toggle";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "./ui/navigation-menu";
import { Sheet, SheetContent, SheetTrigger, SheetClose } from "./ui/sheet";

const navItems = [
  { name: "Home", path: "/" },
  { name: "Abstract", path: "/#abstract" },
  { name: "Results", path: "/#results" },
  { name: "Pipeline", path: "/#pipeline" },
  { name: "Citation", path: "/#citation" },
];

export function Header() {
  return (
    <header className="fixed top-0 z-50 w-full border-b border-border bg-header-background/70 backdrop-blur-md flex justify-center">
      <div className="container mx-auto flex h-16 items-center justify-between px-6 xl:max-w-4xl">
        <div className="flex items-center gap-2">
          <Link to="/" className="flex items-center">
            <span className="font-bold text-lg">LAM3C</span>
          </Link>
        </div>

        <div className="flex items-center gap-4">
          {/* Desktop Navigation */}
          <div className="hidden md:flex">
            <NavigationMenu>
              <NavigationMenuList>
                {navItems.map((item) => (
                  <NavigationMenuItem key={item.path}>
                    <Link to={item.path}>
                      <NavigationMenuLink className="bg-transparent hover:bg-header-accent dark:hover:bg-header-accent/50">
                        {item.name}
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                ))}
              </NavigationMenuList>
            </NavigationMenu>
          </div>

          <ThemeToggle />

          {/* Mobile Navigation */}
          <Sheet>
            <SheetTrigger asChild className="md:hidden">
              <Button variant="ghost" size="icon" aria-label="Menu">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent
              side="right"
              className="bg-header-background/80 backdrop-blur-md"
            >
              <div className="flex flex-col gap-4 py-4">
                {navItems.map((item) => (
                  <SheetClose asChild key={item.path}>
                    <Link
                      to={item.path}
                      className="block px-2 py-1 text-lg font-medium hover:text-primary"
                    >
                      {item.name}
                    </Link>
                  </SheetClose>
                ))}
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}
