library(dplyr)
library(zoo)
library(lubridate)
library(dynlm)
library(CausalImpact)
library(xts)
library(forecast)
library(lmtest)

data <- read.csv("C:/Users/akris/OneDrive/Desktop/Music Worcester Work/anon_DataMerge.csv", stringsAsFactors = FALSE)

data$CreatedDate <- as.Date(data$CreatedDate, format = "%m/%d/%Y")
data$EventDate <- as.Date(data$EventDate, format = "%m/%d/%Y")

filtered_data <- data[
  (data$CreatedDate <= data$EventDate) &
    (format(data$EventDate, "%Y") < 2026) &
    (format(data$CreatedDate, "%Y") <= 2024) &
    (data$Subscriber != TRUE) &
    (data$ChorusMember != TRUE) &
    (data$Student != TRUE) &
    (data$EventType != 'Virtual'),
]

daily_ticket_counts <- filtered_data %>%
  group_by(CreatedDate) %>%
  summarise(TicketCount = n()) %>%
  ungroup()


all_dates <- seq(min(filtered_data$CreatedDate, na.rm = TRUE),
                 max(filtered_data$CreatedDate, na.rm = TRUE), by = "day")

daily_ticket_counts <- data.frame(Date = all_dates) %>%
  left_join(daily_ticket_counts, by = c("Date" = "CreatedDate")) %>%
  mutate(TicketCount = ifelse(is.na(TicketCount), 0, TicketCount))

weekly_ticket_counts <- daily_ticket_counts %>%
  mutate(Week = floor_date(Date, "week")) %>%
  group_by(Week) %>%
  summarise(TicketCount = sum(TicketCount)) %>%
  ungroup()

monthly_ticket_counts <- daily_ticket_counts %>%
  mutate(Month = floor_date(Date, "month")) %>%
  group_by(Month) %>%
  summarise(TicketCount = sum(TicketCount)) %>%
  ungroup()


min_date <- as.Date(min(daily_ticket_counts$Date), format = "%m/%d/%Y")
max_date <- as.Date(max(daily_ticket_counts$Date), format = "%m/%d/%Y")

pre_period <- as.Date(c(min_date, "2020-03-13"))
post_period <- as.Date(c("2020-03-14", "2024-08-12"))

weekly_ticket_counts <- daily_ticket_counts %>%
  mutate(Week = floor_date(Date, "week")) %>%
  group_by(Week) %>%
  summarise(TicketCount = sum(TicketCount)) %>%
  ungroup()

monthly_ticket_counts <- daily_ticket_counts %>%
  mutate(Month = floor_date(Date, "month")) %>%
  group_by(Month) %>%
  summarise(TicketCount = sum(TicketCount)) %>%
  ungroup()

weekly_data <- zoo(weekly_ticket_counts$TicketCount, weekly_ticket_counts$Week)
impact_weekly <- CausalImpact(weekly_data, pre_period, post_period)
summary(impact_weekly)
plot(impact_weekly)
summary(impact_weekly, "report")

monthly_data <- zoo(monthly_ticket_counts$TicketCount, monthly_ticket_counts$Month)
impact_monthly <- CausalImpact(monthly_data, pre_period, post_period)
summary(impact_monthly)
plot(impact_monthly)
summary(impact_monthly, "report")



# data_for_impact <- zoo(daily_ticket_counts$TicketCount, daily_ticket_counts$Date)
# impact <- CausalImpact(data_for_impact, pre_period, post_period)
# summary(impact)
# plot(impact)
# summary(impact, "report")

#-----------------------------------------------------------------------
monthly_ticket_counts <- monthly_ticket_counts %>%
  mutate(MonthNumber = as.numeric(format(Month, "%m")),
         Year = as.numeric(format(Month, "%Y")))

regression_model <- lm(TicketCount ~ MonthNumber + Year, data = monthly_ticket_counts)
summary(regression_model)
