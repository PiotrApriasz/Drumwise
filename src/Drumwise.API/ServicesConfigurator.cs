using System.Reflection;
using Ardalis.GuardClauses;
using Drumwise.Application.Common.Behaviours;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Settings;
using Drumwise.Application.Services;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Data.Interceptors;
using Drumwise.Infrastructure.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using FluentValidation;
using MediatR;
using MediatR.Pipeline;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;

namespace Drumwise.API;

public static class ServicesConfigurator
{
    public static IServiceCollection ConfigureIdentity(this IServiceCollection services, IConfiguration configuration)
    {
        var connectionString = configuration.GetConnectionString("IdentityConnection");
        //Guard.Against.Null(connectionString, $"Connection string for 'IdentityConnection' not found");
        
        services.AddDbContext<AppIdentityDbContext>((sp, options) =>
        {
            options.UseSqlite("DataSource=app.db");
        });
        
        services.AddScoped<IApplicationDbContext>(provider => provider.GetRequiredService<AppIdentityDbContext>());
        services.AddScoped<AppIdentityDbContextInitializer>();

        services.AddIdentityApiEndpoints<ApplicationUser>(options =>
            {
                options.SignIn.RequireConfirmedEmail = true;
            })
            .AddRoles<IdentityRole>()
            .AddEntityFrameworkStores<AppIdentityDbContext>();
        
        services.AddAuthorizationBuilder(); 
        
        services.AddTransient<IIdentityService, IdentityService>();
        services.AddTransient<IEmailSender<ApplicationUser>, IdentityEmailSender>();
        
        services.AddAuthorizationBuilder()
                    .AddPolicy(Policies.CanAddHomework, policy => policy.RequireRole(Roles.Teacher));

        return services;
    }
    
    public static IServiceCollection AddInfrastructuresServices(this IServiceCollection services, IConfiguration configuration)
    {
        var connectionString = configuration.GetConnectionString("AppConnection");
        Guard.Against.Null(connectionString, $"Connection string for 'AppConnection' not found");

        services.AddScoped<ISaveChangesInterceptor, AuditableEntityInterceptor>();
        services.AddScoped<ISaveChangesInterceptor, DispatchDomainEventsInterceptor>();

        services.AddDbContext<ApplicationDbContext>((sp, options) =>
        {
            options.AddInterceptors(sp.GetService<ISaveChangesInterceptor>()!);
            options.UseSqlite(connectionString);
        });

        services.AddScoped<IApplicationDbContext>(provider => provider.GetRequiredService<ApplicationDbContext>());
        services.AddScoped<ApplicationDbContextInitializer>();

        return services;
    }

    public static IServiceCollection AddApplicationServices(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddAutoMapper(Assembly.GetExecutingAssembly());

        services.AddValidatorsFromAssembly(Assembly.GetExecutingAssembly());

        services.AddMediatR(cfg =>
        {
            cfg.RegisterServicesFromAssembly(Assembly.GetExecutingAssembly());
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(UnhandledExceptionBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(AuthorizationBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(PerformanceBahaviour<,>));
            cfg.AddBehavior(typeof(IRequestPreProcessor<>), typeof(LoggingBehaviour<>));
        });

        var smtpSettings = configuration.GetSection("SmtpSettings").Get<SmtpSettings>();
        services.AddTransient<IFileService, FileService>();
        services.AddSingleton(smtpSettings!);

        return services;
    }
}