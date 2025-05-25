import { redirect } from 'react-router';
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from '@/components/ui/navigation-menu';
import { Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

export default function Header() {
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');
  if (theme === 'dark') {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }

  const toggleTheme = () => {
    const htmlElement = document.documentElement;
    if (htmlElement.classList.contains('dark')) {
      htmlElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      setTheme('light');
    } else {
      htmlElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      setTheme('dark');
    }
  };

  return (
    <header className="flex items-center justify-between p-3 border-b">
      <Button
        onClick={() => redirect('/')}
        variant="ghost"
        className="text-lg font-bold"
      >
        DNS 分类系统
      </Button>
      <NavigationMenu>
        <NavigationMenuList>
          <NavigationMenuItem>
            <Button variant="ghost">
              <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                主页
              </NavigationMenuLink>
            </Button>
          </NavigationMenuItem>
          <NavigationMenuItem>
            <Button onClick={toggleTheme} size="icon" variant="ghost">
              {theme === 'dark' ? (
                <Sun className="size-5" />
              ) : (
                <Moon className="size-5" />
              )}
            </Button>
          </NavigationMenuItem>
        </NavigationMenuList>
      </NavigationMenu>
    </header>
  );
}
