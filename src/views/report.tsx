import Header from '@/components/Header';
import { Card } from '@/components/ui/card';
import { useParams } from 'react-router';
import { Skeleton } from '@/components/ui/skeleton';
import { TrendingUp } from 'lucide-react';
import { Label, Pie, PieChart, ResponsiveContainer } from 'recharts';
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import { useEffect, useState } from 'react';

export function Report() {
  const params = useParams<{ reportId: string }>();

  const chartData = [
    { browser: 'Chrome', visitors: 275, fill: 'var(--chart-1)' },
    { browser: 'Safari', visitors: 200, fill: 'var(--chart-2)' },
  ];

  const chartConfig = {
    values: {
      label: 'Values',
    },
  } satisfies ChartConfig;

  const [loaded, setLoaded] = useState<boolean>(false);
  useEffect(() => {
    
  }, []);

  return (
    <div className="bg-background text-foreground h-screen w-screen">
      <Header />
      <main className="flex flex-col h-full w-full p-4">
        <div className="flex flex-col items-center justify-center">
          <h1 className="text-2xl font-bold mb-4">检查报告</h1>
          <p className="text-gray-600">报告ID: {params.reportId}</p>
        </div>
        <div className="flex flex-1 w-full">
          <Card className="w-full max-w-3xl mx-auto my-4 p-6">
            <h2 className="text-xl font-semibold mb-4">报告内容</h2>
            {!loaded && (
              <>
                <Skeleton className="w-[300px] h-[300px] rounded-3xl self-center" />
                <Skeleton className="w-full h-[30px] rounded-md" />
                <Skeleton className="w-full h-[30px] rounded-md" />
                <Skeleton className="w-[30%] h-[30px] rounded-md" />
              </>
            )}
            {loaded && (
              <>
                <ChartContainer
                  config={chartConfig}
                  className="aspect-square max-h-[250px]"
                >
                  <PieChart>
                    <ChartTooltip
                      cursor={false}
                      content={<ChartTooltipContent hideLabel />}
                    />
                    <Pie
                      data={chartData}
                      dataKey="visitors"
                      nameKey="browser"
                      innerRadius={60}
                      strokeWidth={5}
                    >
                      <Label
                        content={({ viewBox }) => {
                          if (viewBox && 'cx' in viewBox && 'cy' in viewBox) {
                            return (
                              <text
                                x={viewBox.cx}
                                y={viewBox.cy}
                                textAnchor="middle"
                                dominantBaseline="middle"
                              >
                                <tspan
                                  x={viewBox.cx}
                                  y={viewBox.cy}
                                  className="fill-foreground text-3xl font-bold"
                                >
                                  {'000'}
                                </tspan>
                                <tspan
                                  x={viewBox.cx}
                                  y={(viewBox.cy || 0) + 24}
                                  className="fill-muted-foreground"
                                >
                                  Visitors
                                </tspan>
                              </text>
                            );
                          }
                        }}
                      />
                    </Pie>
                  </PieChart>
                </ChartContainer>
                sss
              </>
            )}
          </Card>
        </div>
      </main>
      <footer className="text-center text-sm text-gray-500 p-4">
        © 2023 DNS 分类系统
      </footer>
    </div>
  );
}
